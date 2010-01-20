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
package org.apache.mahout.clustering.dirichlet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.dirichlet.models.L1ModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.lucene.CachedTermInfo;
import org.apache.mahout.utils.vectors.lucene.LuceneIterable;
import org.apache.mahout.utils.vectors.lucene.TFDFMapper;
import org.apache.mahout.utils.vectors.lucene.VectorMapper;
import org.junit.After;
import org.junit.Before;

public class TestL1ModelClustering extends MahoutTestCase {

  private static final String[] DOCS = { "The quick red fox jumped over the lazy brown dogs.",
      "The quick brown fox jumped over the lazy red dogs.", "The quick red cat jumped over the lazy brown dogs.",
      "The quick brown cat jumped over the lazy red dogs.", "Mary had a little lamb whose fleece was white as snow.",
      "Moby Dick is a story of a whale and a man obsessed.", "The robber wore a black fleece jacket and a baseball cap.",
      "The English Springer Spaniel is the best of all dogs." };

  private List<VectorWritable> sampleData;

  Random random;

  private RAMDirectory directory;

  @Before
  protected void setUp() throws Exception {
    super.setUp();
    random = RandomUtils.getRandom();
    sampleData = new ArrayList<VectorWritable>();

    directory = new RAMDirectory();
    IndexWriter writer = new IndexWriter(directory, new StandardAnalyzer(Version.LUCENE_CURRENT), true,
        IndexWriter.MaxFieldLength.UNLIMITED);
    for (int i = 0; i < DOCS.length; i++) {
      Document doc = new Document();
      Field id = new Field("id", "doc_" + i, Field.Store.YES, Field.Index.NOT_ANALYZED_NO_NORMS);
      doc.add(id);
      //Store both position and offset information
      Field text = new Field("content", DOCS[i], Field.Store.NO, Field.Index.ANALYZED, Field.TermVector.YES);
      doc.add(text);
      writer.addDocument(doc);
    }
    writer.close();
  }

  @After
  protected void tearDown() throws Exception {
  }

  private static String formatVector(Vector v) {
    StringBuilder buf = new StringBuilder();
    int nzero = 0;
    Iterator<Element> iterateNonZero = v.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      iterateNonZero.next();
      nzero++;
    }
    buf.append("(").append(nzero);
    buf.append("nz) [");
    int nextIx = 0;
    if (v != null) {
      // handle sparse Vectors gracefully, suppressing zero values
      for (int i = 0; i < v.size(); i++) {
        double elem = v.get(i);
        if (elem == 0.0)
          continue;
        if (i > nextIx)
          buf.append("..{").append(i).append("}=");
        buf.append(String.format("%.2f", elem)).append(", ");
        nextIx = i + 1;
      }
    }
    buf.append("]");
    return buf.toString();

  }

  private static void printResults(List<Model<VectorWritable>[]> result, int significant) {
    int row = 0;
    for (Model<VectorWritable>[] r : result) {
      int sig = 0;
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          sig++;
        }
      }
      System.out.print("sample[" + row++ + "] (" + sig + ")= ");
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          System.out.print(model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  public void testDocs() throws Exception {
    IndexReader reader = IndexReader.open(directory, true);
    Weight weight = new TFIDF();
    TermInfo termInfo = new CachedTermInfo(reader, "content", 1, 100);
    VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
    LuceneIterable iterable = new LuceneIterable(reader, "id", "content", mapper);

    for (Vector vector : iterable) {
      assertNotNull(vector);
      System.out.println("Vector=" + formatVector(vector));
      sampleData.add(new VectorWritable(vector));
    }
    DirichletClusterer<VectorWritable> dc = new DirichletClusterer<VectorWritable>(sampleData, new L1ModelDistribution(sampleData
        .get(0)), 1.0, 15, 1, 0);
    List<Model<VectorWritable>[]> result = dc.cluster(10);
    printResults(result, 0);
    assertNotNull(result);

  }

}
