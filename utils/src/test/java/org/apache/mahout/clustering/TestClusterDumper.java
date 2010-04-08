/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.canopy.CanopyClusteringJob;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.dirichlet.DirichletClusterer;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.models.L1ModelDistribution;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.lucene.CachedTermInfo;
import org.apache.mahout.utils.vectors.lucene.LuceneIterable;
import org.apache.mahout.utils.vectors.lucene.TFDFMapper;
import org.apache.mahout.utils.vectors.lucene.VectorMapper;

public class TestClusterDumper extends MahoutTestCase {

  private List<VectorWritable> sampleData;

  private FileSystem fs;

  private static final String[] DOCS = { "The quick red fox jumped over the lazy brown dogs.",
      "The quick brown fox jumped over the lazy red dogs.", "The quick red cat jumped over the lazy brown dogs.",
      "The quick brown cat jumped over the lazy red dogs.", "Mary had a little lamb whose fleece was white as snow.",
      "Mary had a little goat whose fleece was white as snow.",
      "Mary had a little lamb whose fleece was black as tar.",
      "Dick had a little goat whose fleece was white as snow.", "Moby Dick is a story of a whale and a man obsessed.",
      "Moby Bob is a story of a walrus and a man obsessed.", "Moby Dick is a story of a whale and a crazy man.",
      "The robber wore a black fleece jacket and a baseball cap.",
      "The robber wore a red fleece jacket and a baseball cap.",
      "The robber wore a white fleece jacket and a baseball cap.",
      "The English Springer Spaniel is the best of all dogs." };

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
    // Create testdata directory
    File f = new File("testdata");
    if (!f.exists()) {
      f.mkdir();
    }
    f = new File("testdata/points");
    if (!f.exists()) {
      f.mkdir();
    }
    f = new File("output");
    rmDir(f);
    // Create test data
    getSampleData(DOCS);
    ClusteringTestUtils.writePointsToFile(sampleData, "testdata/points/file1", fs, conf);
    // Run clustering job
    // Run ClusterDumper test
  }

  private void rmDir(File f) {
    if (f != null && f.exists()) {
      if (f.isDirectory())
        for (File g : f.listFiles())
          rmDir(g);
      f.delete();
    }
  }

  private void getSampleData(String[] docs2) throws IOException {
    sampleData = new ArrayList<VectorWritable>();
    RAMDirectory directory = new RAMDirectory();
    IndexWriter writer = new IndexWriter(directory, new StandardAnalyzer(Version.LUCENE_CURRENT), true,
        IndexWriter.MaxFieldLength.UNLIMITED);
    for (int i = 0; i < docs2.length; i++) {
      Document doc = new Document();
      Field id = new Field("id", "doc_" + i, Field.Store.YES, Field.Index.NOT_ANALYZED_NO_NORMS);
      doc.add(id);
      // Store both position and offset information
      Field text = new Field("content", docs2[i], Field.Store.NO, Field.Index.ANALYZED, Field.TermVector.YES);
      doc.add(text);
      writer.addDocument(doc);
    }
    writer.close();
    IndexReader reader = IndexReader.open(directory, true);
    Weight weight = new TFIDF();
    TermInfo termInfo = new CachedTermInfo(reader, "content", 1, 100);
    VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
    LuceneIterable iterable = new LuceneIterable(reader, "id", "content", mapper);

    int i = 0;
    for (Vector vector : iterable) {
      Assert.assertNotNull(vector);
      System.out.println("Vector[" + i++ + "]=" + ClusterBase.formatVector(vector, null));
      sampleData.add(new VectorWritable(vector));
    }
  }

  public void testCanopy() throws Exception { // now run the Job
    CanopyClusteringJob.runJob("testdata/points", "output", EuclideanDistanceMeasure.class.getName(), 8, 4);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper("output/canopies", null);
    clusterDumper.printClusters();
  }

  public void testKmeans() throws Exception {
    // now run the Canopy job to prime kMeans canopies
    CanopyDriver.runJob("testdata/points", "testdata/canopies", EuclideanDistanceMeasure.class.getName(), 8, 4);
    // now run the KMeans job
    KMeansDriver.runJob("testdata/points", "testdata/canopies", "output", EuclideanDistanceMeasure.class.getName(),
        0.001, 10, 1);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper("output/clusters-1", null);
    clusterDumper.printClusters();
  }

  public void testDirichlet() throws Exception {
    Vector prototype = sampleData.get(0).get();
    DirichletDriver.runJob("testdata/points", "output",
        new L1ModelDistribution(sampleData.get(0)).getClass().getName(), prototype.getClass().getName(), prototype
            .size(), 15, 10, 1.0, 1);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper("output/state-10", null);
    clusterDumper.printClusters();
  }
}
