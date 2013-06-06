package org.apache.mahout.text;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.Pair;
import org.apache.mahout.text.doc.MultipleFieldsDocument;
import org.apache.mahout.text.doc.NumericFieldDocument;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.apache.mahout.vectorizer.DefaultAnalyzer;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Abstract test for working with Lucene storage.
 */
public abstract class AbstractLuceneStorageTest {

  private Path indexPath = new Path("index");

  protected void commitDocuments(SingleFieldDocument... documents) throws IOException {
    IndexWriter indexWriter = new IndexWriter(getDirectory(), new IndexWriterConfig(Version.LUCENE_35, new DefaultAnalyzer()));

    for (SingleFieldDocument singleFieldDocument : documents) {
      indexWriter.addDocument(singleFieldDocument.asLuceneDocument());
    }

    indexWriter.commit();
    indexWriter.close();
  }

  protected void assertSimpleDocumentEquals(SingleFieldDocument expected, Pair<Text, Text> actual) {
    assertEquals(expected.getId(), actual.getFirst().toString());
    assertEquals(expected.getField(), actual.getSecond().toString());
  }

  protected void assertMultipleFieldsDocumentEquals(MultipleFieldsDocument expected, Pair<Text, Text> actual) {
    assertEquals(expected.getId(), actual.getFirst().toString());
    assertEquals(expected.getField() + " " + expected.getField1() + " " + expected.getField2(), actual.getSecond().toString());
  }

  protected void assertNumericFieldEquals(NumericFieldDocument expected, Pair<Text, Text> actual) {
    assertEquals(expected.getId(), actual.getFirst().toString());
    assertEquals(expected.getField() + " " + expected.getNumericField(), actual.getSecond().toString());
  }

  protected FSDirectory getDirectory() throws IOException {
    return FSDirectory.open(new File(indexPath.toString()));
  }

  protected Path getIndexPath() {
    return indexPath;
  }
}
