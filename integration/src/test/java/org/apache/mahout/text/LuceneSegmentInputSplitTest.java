package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.apache.mahout.vectorizer.DefaultAnalyzer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static java.util.Arrays.asList;
import static junit.framework.Assert.assertEquals;

public class LuceneSegmentInputSplitTest {

  private FSDirectory directory;
  private Path indexPath;
  private Configuration conf;

  @Before
  public void before() throws IOException {
    indexPath = new Path("index");
    directory = FSDirectory.open(new File(indexPath.toString()));
    conf = new Configuration();
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(conf, indexPath);
  }

  @Test
  public void testGetSegment() throws Exception {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    List<SingleFieldDocument> docs = asList(doc1, doc2, doc3);
    for (SingleFieldDocument doc : docs) {
      addDocument(doc);
    }

    assertSegmentContainsOneDoc("_0");
    assertSegmentContainsOneDoc("_1");
    assertSegmentContainsOneDoc("_2");
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetSegment_nonExistingSegment() throws Exception {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    List<SingleFieldDocument> docs = asList(doc1, doc2, doc3);
    for (SingleFieldDocument doc : docs) {
      addDocument(doc);
    }

    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath, "_3", 1000);
    inputSplit.getSegment(conf);
  }

  private void assertSegmentContainsOneDoc(String segmentName) throws IOException {
    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath, segmentName, 1000);
    SegmentInfo segment = inputSplit.getSegment(conf);
    SegmentReader segmentReader = SegmentReader.get(true, segment, 1);
    assertEquals(segmentName, segment.name);
    assertEquals(1, segmentReader.numDocs());
  }

  private void addDocument(SingleFieldDocument doc) throws IOException {
    IndexWriterConfig conf = new IndexWriterConfig(Version.LUCENE_35, new DefaultAnalyzer());
    IndexWriter indexWriter = new IndexWriter(directory, conf);
    indexWriter.addDocument(doc.asLuceneDocument());
    indexWriter.commit();
    indexWriter.close();
  }
}
