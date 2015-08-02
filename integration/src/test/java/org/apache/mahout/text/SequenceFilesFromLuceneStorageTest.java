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
package org.apache.mahout.text;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.text.doc.MultipleFieldsDocument;
import org.apache.mahout.text.doc.NumericFieldDocument;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.apache.mahout.text.doc.TestDocument;
import org.apache.mahout.text.doc.UnstoredFieldsDocument;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
@Deprecated
public class SequenceFilesFromLuceneStorageTest extends AbstractLuceneStorageTest {

  private SequenceFilesFromLuceneStorage lucene2Seq;
  private LuceneStorageConfiguration lucene2SeqConf;
  private Path seqFilesOutputPath;
  private Configuration configuration;

  @Before
  public void before() throws IOException {
    configuration = getConfiguration();
    seqFilesOutputPath = new Path(getTestTempDirPath(), "seqfiles");

    lucene2Seq = new SequenceFilesFromLuceneStorage();
    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      Arrays.asList(getIndexPath1(), getIndexPath2()), seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD, Collections.singletonList(SingleFieldDocument.FIELD));
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getSequenceFilesOutputPath());
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getIndexPaths());
  }

  @Test
  public void testRun2Directories() throws Exception {
    //Two commit points, each in two diff. Directories
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(1000, 1500));

    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(500, 1000));
    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(1500, 2000));

    commitDocuments(getDirectory(getIndexPath1AsFile()), misshapenDocs);
    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();
    Map<String, Text> map = new HashMap<>();
    while (iterator.hasNext()) {
      Pair<Text, Text> next = iterator.next();
      map.put(next.getFirst().toString(), next.getSecond());
    }
    assertEquals(docs.size() + misshapenDocs.size(), map.size());
    for (TestDocument doc : docs) {
      Text value = map.get(doc.getId());
      assertNotNull(value);
      assertEquals(value.toString(), doc.getField());
    }
    for (TestDocument doc : misshapenDocs) {
      Text value = map.get(doc.getId());
      assertNotNull(value);
      assertEquals(value.toString(), doc.getField());
    }
  }

  @Test(expected = IllegalArgumentException.class)
  public void testRunUnstoredFields() throws IOException {
    commitDocuments(getDirectory(getIndexPath1AsFile()), new UnstoredFieldsDocument("5", "This is test document 5"));

    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      Collections.singletonList(getIndexPath1()), seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD, Arrays.asList(UnstoredFieldsDocument.FIELD, UnstoredFieldsDocument.UNSTORED_FIELD));

    lucene2Seq.run(lucene2SeqConf);
  }

  @Test
  public void testRunMaxHits() throws IOException {
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(1000, 1500));

    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(500, 1000));
    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(1500, 2000));

    lucene2SeqConf.setMaxHits(3);
    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();
    assertTrue(iterator.hasNext());
    iterator.next();
    assertTrue(iterator.hasNext());
    iterator.next();
    assertTrue(iterator.hasNext());
    iterator.next();
    assertFalse(iterator.hasNext());
  }

  @Test
  public void testRunQuery() throws IOException {
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs);
    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      Collections.singletonList(getIndexPath1()), seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD, Collections.singletonList(SingleFieldDocument.FIELD));

    Query query = new TermQuery(new Term(lucene2SeqConf.getFields().get(0), "599"));

    lucene2SeqConf.setQuery(query);
    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();
    assertTrue(iterator.hasNext());
    Pair<Text, Text> next = iterator.next();
    assertTrue(next.getSecond().toString().contains("599"));
    assertFalse(iterator.hasNext());
  }

  @Test
  public void testRunMultipleFields() throws IOException {
    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      Collections.singletonList(getIndexPath1()), seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD,
      Arrays.asList(MultipleFieldsDocument.FIELD, MultipleFieldsDocument.FIELD1, MultipleFieldsDocument.FIELD2));

    MultipleFieldsDocument multipleFieldsDocument1 =
        new MultipleFieldsDocument("1", "This is field 1-1", "This is field 1-2", "This is field 1-3");
    MultipleFieldsDocument multipleFieldsDocument2 =
        new MultipleFieldsDocument("2", "This is field 2-1", "This is field 2-2", "This is field 2-3");
    MultipleFieldsDocument multipleFieldsDocument3 =
        new MultipleFieldsDocument("3", "This is field 3-1", "This is field 3-2", "This is field 3-3");
    commitDocuments(getDirectory(getIndexPath1AsFile()), multipleFieldsDocument1,
        multipleFieldsDocument2, multipleFieldsDocument3);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertMultipleFieldsDocumentEquals(multipleFieldsDocument1, iterator.next());
    assertMultipleFieldsDocumentEquals(multipleFieldsDocument2, iterator.next());
    assertMultipleFieldsDocumentEquals(multipleFieldsDocument3, iterator.next());
  }

  @Test
  public void testRunNumericField() throws IOException {
    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      Collections.singletonList(getIndexPath1()), seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD, Arrays.asList(NumericFieldDocument.FIELD, NumericFieldDocument.NUMERIC_FIELD));

    NumericFieldDocument doc1 = new NumericFieldDocument("1", "This is field 1", 100);
    NumericFieldDocument doc2 = new NumericFieldDocument("2", "This is field 2", 200);
    NumericFieldDocument doc3 = new NumericFieldDocument("3", "This is field 3", 300);

    commitDocuments(getDirectory(getIndexPath1AsFile()), doc1, doc2, doc3);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertNumericFieldEquals(doc1, iterator.next());
    assertNumericFieldEquals(doc2, iterator.next());
    assertNumericFieldEquals(doc3, iterator.next());
  }

  @Test(expected = IllegalArgumentException.class)
  public void testNonExistingIdField() throws IOException {
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));

    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
        Collections.singletonList(getIndexPath1()),
        seqFilesOutputPath,
        "nonExistingField",
        Collections.singletonList(SingleFieldDocument.FIELD));

    lucene2Seq.run(lucene2SeqConf);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testNonExistingField() throws IOException {
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));

    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
        Collections.singletonList(getIndexPath1()),
        seqFilesOutputPath,
        SingleFieldDocument.ID_FIELD,
        Arrays.asList(SingleFieldDocument.FIELD, "nonExistingField"));

    lucene2Seq.run(lucene2SeqConf);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testIndexedButNotStoredField() throws IOException {
    SingleFieldDocument document = new SingleFieldDocument("id", "field") {
      @Override
      public Document asLuceneDocument() {
        Document document = super.asLuceneDocument();
        document.add(new TextField("indexed", "This text is indexed", Field.Store.NO));
        return document;
      }
    };
    commitDocuments(getDirectory(getIndexPath1AsFile()), document);

    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
        Collections.singletonList(getIndexPath1()),
        seqFilesOutputPath,
        SingleFieldDocument.ID_FIELD,
        Arrays.asList(SingleFieldDocument.FIELD, "indexed"));

    lucene2Seq.run(lucene2SeqConf);
  }
}
