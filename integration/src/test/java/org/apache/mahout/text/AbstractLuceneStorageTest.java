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

import com.google.common.collect.Lists;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.text.doc.MultipleFieldsDocument;
import org.apache.mahout.text.doc.NumericFieldDocument;
import org.apache.mahout.text.doc.SingleFieldDocument;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Abstract test for working with Lucene storage.
 */
public abstract class AbstractLuceneStorageTest extends MahoutTestCase {

  protected Path indexPath1;
  protected Path indexPath2;
  protected List<SingleFieldDocument> docs = Lists.newArrayList();
  protected List<SingleFieldDocument> misshapenDocs = Lists.newArrayList();

  @Override
  public void setUp() throws Exception {
    super.setUp();
    indexPath1 = getTestTempDirPath("index1");
    indexPath2 = getTestTempDirPath("index2");
    for (int i = 0; i < 2000; i++) {
      docs.add(new SingleFieldDocument(String.valueOf(i), "This is test document " + i));
    }
    misshapenDocs.add(new SingleFieldDocument("", "This doc has an empty id"));
    misshapenDocs.add(new SingleFieldDocument("empty_value", ""));
  }

  protected void commitDocuments(Directory directory, Iterable<SingleFieldDocument> theDocs) throws IOException{
    IndexWriter indexWriter = new IndexWriter(directory, new IndexWriterConfig(Version.LUCENE_46, new StandardAnalyzer(Version.LUCENE_46)));

    for (SingleFieldDocument singleFieldDocument : theDocs) {
      indexWriter.addDocument(singleFieldDocument.asLuceneDocument());
    }

    indexWriter.commit();
    indexWriter.close();
  }

  protected void commitDocuments(Directory directory, SingleFieldDocument... documents) throws IOException {
    commitDocuments(directory, Arrays.asList(documents));
  }

  protected void assertMultipleFieldsDocumentEquals(MultipleFieldsDocument expected, Pair<Text, Text> actual) {
    assertEquals(expected.getId(), actual.getFirst().toString());
    assertEquals(expected.getField() + " " + expected.getField1() + " " + expected.getField2(), actual.getSecond().toString());
  }

  protected void assertNumericFieldEquals(NumericFieldDocument expected, Pair<Text, Text> actual) {
    assertEquals(expected.getId(), actual.getFirst().toString());
    assertEquals(expected.getField() + " " + expected.getNumericField(), actual.getSecond().toString());
  }

  protected FSDirectory getDirectory(File indexPath) throws IOException {
    return FSDirectory.open(indexPath);
  }

  protected File getIndexPath1AsFile() {
    return new File(indexPath1.toUri().getPath());
  }

  protected Path getIndexPath1() {
    return indexPath1;
  }

  protected File getIndexPath2AsFile() {
    return new File(indexPath2.toUri().getPath());
  }

  protected Path getIndexPath2() {
    return indexPath2;
  }
}
