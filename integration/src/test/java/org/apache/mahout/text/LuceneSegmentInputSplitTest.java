package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.lucene.index.SegmentInfoPerCommit;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static java.util.Arrays.asList;

public class LuceneSegmentInputSplitTest extends AbstractLuceneStorageTest {

  private FSDirectory directory;

  private Configuration conf;

  @Before
  public void before() throws IOException {
    directory = getDirectory(getIndexPath1AsFile());
    conf = new Configuration();
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(conf, indexPath1);
  }

  @Test
  public void testGetSegment() throws Exception {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    List<SingleFieldDocument> docs = asList(doc1, doc2, doc3);
    for (SingleFieldDocument doc : docs) {
      commitDocuments(getDirectory(getIndexPath1AsFile()), doc);
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
      commitDocuments(getDirectory(getIndexPath1AsFile()), doc);
    }

    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath1, "_3", 1000);
    inputSplit.getSegment(conf);
  }

  private void assertSegmentContainsOneDoc(String segmentName) throws IOException {
    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath1, segmentName, 1000);
    SegmentInfoPerCommit segment = inputSplit.getSegment(conf);
    SegmentReader segmentReader = new SegmentReader(segment, 1, IOContext.READ);//SegmentReader.get(true, segment, 1);
    assertEquals(segmentName, segment.info.name);
    assertEquals(1, segmentReader.numDocs());
  }


}
