package org.apache.mahout.utils.vectors.lucene;
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

import junit.framework.TestCase;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Document;
import org.apache.lucene.util.Version;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.SparseVector;

/**
 *
 *
 **/
public class LuceneIterableTest extends TestCase {
  protected RAMDirectory directory;

  private static final String [] DOCS = {
        "The quick red fox jumped over the lazy brown dogs.",
        "Mary had a little lamb whose fleece was white as snow.",
        "Moby Dick is a story of a whale and a man obsessed.",
        "The robber wore a black fleece jacket and a baseball cap.",
        "The English Springer Spaniel is the best of all dogs."
    };


  @Override
  protected void setUp() throws Exception {
    directory = new RAMDirectory();
    IndexWriter writer = new IndexWriter(directory, new StandardAnalyzer(Version.LUCENE_CURRENT), true, IndexWriter.MaxFieldLength.UNLIMITED);
    for (int i = 0; i < DOCS.length; i++){
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

  public void testIterable() throws Exception {
    IndexReader reader = IndexReader.open(directory, true);
    Weight weight = new TFIDF();
    TermInfo termInfo = new CachedTermInfo(reader, "content", 1, 100);
    VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
    LuceneIterable iterable = new LuceneIterable(reader, "id", "content", mapper);

    //TODO: do something more meaningful here
    for (Vector vector : iterable) {
      assertTrue("vector is not an instanceof " + SparseVector.class, vector instanceof SparseVector);
      assertTrue("vector Size: " + vector.size() + " is not greater than: " + 0, vector.size() > 0);
    }
  }


}
