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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.lucene.CachedTermInfo;
import org.apache.mahout.utils.vectors.lucene.LuceneIterable;
import org.apache.mahout.vectorizer.TFIDF;
import org.apache.mahout.vectorizer.Weight;
import org.junit.Before;
import org.junit.Test;

public final class TestClusterDumper extends MahoutTestCase {

  private static final String[] DOCS = {
      "The quick red fox jumped over the lazy brown dogs.",
      "The quick brown fox jumped over the lazy red dogs.",
      "The quick red cat jumped over the lazy brown dogs.",
      "The quick brown cat jumped over the lazy red dogs.",
      "Mary had a little lamb whose fleece was white as snow.",
      "Mary had a little goat whose fleece was white as snow.",
      "Mary had a little lamb whose fleece was black as tar.",
      "Dick had a little goat whose fleece was white as snow.",
      "Moby Dick is a story of a whale and a man obsessed.",
      "Moby Bob is a story of a walrus and a man obsessed.",
      "Moby Dick is a story of a whale and a crazy man.",
      "The robber wore a black fleece jacket and a baseball cap.",
      "The robber wore a red fleece jacket and a baseball cap.",
      "The robber wore a white fleece jacket and a baseball cap.",
      "The English Springer Spaniel is the best of all dogs."};

  private List<VectorWritable> sampleData;

  private String[] termDictionary;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);
    // Create test data
    getSampleData(DOCS);
    ClusteringTestUtils.writePointsToFile(sampleData, true,
        getTestTempFilePath("testdata/file1"), fs, conf);
  }

  private void getSampleData(String[] docs2) throws IOException {
    sampleData = new ArrayList<>();
    RAMDirectory directory = new RAMDirectory();
    try (IndexWriter writer = new IndexWriter(directory,
        new IndexWriterConfig(Version.LUCENE_46, new StandardAnalyzer(Version.LUCENE_46)))){
      for (int i = 0; i < docs2.length; i++) {
        Document doc = new Document();
        Field id = new StringField("id", "doc_" + i, Field.Store.YES);
        doc.add(id);
        // Store both position and offset information
        FieldType fieldType = new FieldType();
        fieldType.setStored(false);
        fieldType.setIndexed(true);
        fieldType.setTokenized(true);
        fieldType.setStoreTermVectors(true);
        fieldType.setStoreTermVectorPositions(true);
        fieldType.setStoreTermVectorOffsets(true);
        fieldType.freeze();
        Field text = new Field("content", docs2[i], fieldType);
        doc.add(text);
        writer.addDocument(doc);
      }
    }

    IndexReader reader = DirectoryReader.open(directory);

    Weight weight = new TFIDF();
    TermInfo termInfo = new CachedTermInfo(reader, "content", 1, 100);

    int numTerms = 0;
    for (Iterator<TermEntry> it = termInfo.getAllEntries(); it.hasNext();) {
      it.next();
      numTerms++;
    }
    termDictionary = new String[numTerms];
    int i = 0;
    for (Iterator<TermEntry> it = termInfo.getAllEntries(); it.hasNext();) {
      String term = it.next().getTerm();
      termDictionary[i] = term;
      System.out.println(i + " " + term);
      i++;
    }
    Iterable<Vector> iterable = new LuceneIterable(reader, "id", "content",
        termInfo,weight);

    i = 0;
    for (Vector vector : iterable) {
      assertNotNull(vector);
      NamedVector namedVector;
      if (vector instanceof NamedVector) {
        // rename it for testing purposes
        namedVector = new NamedVector(((NamedVector) vector).getDelegate(),
            "P(" + i + ')');

      } else {
        namedVector = new NamedVector(vector, "P(" + i + ')');
      }
      System.out.println(AbstractCluster.formatVector(namedVector,
          termDictionary));
      sampleData.add(new VectorWritable(namedVector));
      i++;
    }
  }

  /**
   * Return the path to the final iteration's clusters
   */
  private static Path finalClusterPath(Configuration conf, Path output,
      int maxIterations) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    for (int i = maxIterations; i >= 0; i--) {
      Path clusters = new Path(output, "clusters-" + i + "-final");
      if (fs.exists(clusters)) {
        return clusters;
      }
    }
    return null;
  }

  @Test
  public void testKmeans() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Path input = getTestTempFilePath("input");
    Path output = getTestTempDirPath("output");
    Path initialPoints = new Path(output, Cluster.CLUSTERS_DIR + '0' + Cluster.FINAL_ITERATION_SUFFIX);
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);
    // Write test data to file
    ClusteringTestUtils.writePointsToFile(sampleData, input, fs, conf);
    // Select initial centroids
    RandomSeedGenerator.buildRandom(conf, input, initialPoints, 8, measure, 1L);
    // Run k-means
    Path kMeansOutput = new Path(output, "kmeans");
    KMeansDriver.run(conf, getTestTempDirPath("testdata"), initialPoints, kMeansOutput, 0.001, 10, true, 0.0, false);
    // Print out clusters
    ClusterDumper clusterDumper = new ClusterDumper(finalClusterPath(conf,
            output, 10), new Path(kMeansOutput, "clusteredPoints"));
    clusterDumper.printClusters(termDictionary);
  }

  @Test
  public void testJsonClusterDumper() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Path input = getTestTempFilePath("input");
    Path output = getTestTempDirPath("output");
    Path initialPoints = new Path(output, Cluster.CLUSTERS_DIR + '0' + Cluster.FINAL_ITERATION_SUFFIX);
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);
    // Write test data to file
    ClusteringTestUtils.writePointsToFile(sampleData, input, fs, conf);
    // Select initial centroids
    RandomSeedGenerator.buildRandom(conf, input, initialPoints, 8, measure, 1L);
    // Run k-means
    Path kmeansOutput = new Path(output, "kmeans");
    KMeansDriver.run(conf, getTestTempDirPath("testdata"), initialPoints, kmeansOutput, 0.001, 10, true, 0.0, false);
    // Print out clusters
    ClusterDumper clusterDumper = new ClusterDumper(finalClusterPath(conf,
        output, 10), new Path(kmeansOutput, "clusteredPoints"));
    clusterDumper.setOutputFormat(ClusterDumper.OUTPUT_FORMAT.JSON);
    clusterDumper.printClusters(termDictionary);
  }

  @Test
  public void testFuzzyKmeans() throws Exception {
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    Path input = getTestTempFilePath("input");
    Path output = getTestTempDirPath("output");
    Path initialPoints = new Path(output, Cluster.CLUSTERS_DIR + '0' + Cluster.FINAL_ITERATION_SUFFIX);
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);
    // Write test data to file
    ClusteringTestUtils.writePointsToFile(sampleData, input, fs, conf);
    // Select initial centroids
    RandomSeedGenerator.buildRandom(conf, input, initialPoints, 8, measure, 1L);
    // Run k-means
    Path kMeansOutput = new Path(output, "kmeans");
    FuzzyKMeansDriver.run(conf, getTestTempDirPath("testdata"), initialPoints, kMeansOutput, 0.001, 10, 1.1f, true,
        true, 0, true);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(finalClusterPath(conf,
        output, 10), new Path(kMeansOutput, "clusteredPoints"));
    clusterDumper.printClusters(termDictionary);
  }
}
