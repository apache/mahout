package org.apache.mahout.utils.vectors.lucene;


import com.google.common.io.Closeables;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.utils.MahoutTestCase;
import org.junit.Test;

import java.io.IOException;

/**
 *
 *
 **/
public class CachedTermInfoTest extends MahoutTestCase {
  private RAMDirectory directory;
  private static final String[] DOCS = {
          "a a b b c c",
          "a b a b a b a b",
          "a b a",
          "a",
          "b",
          "a",
          "a"
  };

  private static final String[] DOCS2 = {
          "d d d d",
          "e e e e",
          "d e d e",
          "d",
          "e",
          "d",
          "e"
  };

  @Override
  public void setUp() throws Exception {
    super.setUp();
    directory = new RAMDirectory();
    directory = createTestIndex(Field.TermVector.NO, directory, true, 0);
  }

  @Test
  public void test() throws Exception {
    IndexReader reader = DirectoryReader.open(directory);
    CachedTermInfo cti = new CachedTermInfo(reader, "content", 0, 100);
    assertEquals(3, cti.totalTerms("content"));
    assertNotNull(cti.getTermEntry("content", "a"));
    assertNull(cti.getTermEntry("content", "e"));
    //minDf
    cti = new CachedTermInfo(reader, "content", 3, 100);
    assertEquals(2, cti.totalTerms("content"));
    assertNotNull(cti.getTermEntry("content", "a"));
    assertNull(cti.getTermEntry("content", "c"));
    //maxDFPercent, a is in 6 of 7 docs: numDocs * maxDfPercent / 100 < 6 to exclude, 85% should suffice to exclude a
    cti = new CachedTermInfo(reader, "content", 0, 85);
    assertEquals(2, cti.totalTerms("content"));
    assertNotNull(cti.getTermEntry("content", "b"));
    assertNotNull(cti.getTermEntry("content", "c"));
    assertNull(cti.getTermEntry("content", "a"));


  }

  static RAMDirectory createTestIndex(Field.TermVector termVector,
                                      RAMDirectory directory,
                                      boolean createNew,
                                      int startingId) throws IOException {
    IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(Version.LUCENE_41, new WhitespaceAnalyzer(Version.LUCENE_41)));

    try {
      for (int i = 0; i < DOCS.length; i++) {
        Document doc = new Document();
        Field id = new StringField("id", "doc_" + (i + startingId), Field.Store.YES);
        doc.add(id);
        //Store both position and offset information
        //Says it is deprecated, but doesn't seem to offer an alternative that supports term vectors...
        Field text = new Field("content", DOCS[i], Field.Store.NO, Field.Index.ANALYZED, termVector);
        doc.add(text);
        Field text2 = new Field("content2", DOCS2[i], Field.Store.NO, Field.Index.ANALYZED, termVector);
        doc.add(text2);
        writer.addDocument(doc);
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    return directory;
  }
}
