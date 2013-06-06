package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.lucene.store.FSDirectory;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class LuceneSegmentInputFormatTest extends AbstractLuceneStorageTest {

  private LuceneSegmentInputFormat inputFormat;
  private JobContext jobContext;
  private Configuration conf;

  @Before
  public void before() throws IOException {
    inputFormat = new LuceneSegmentInputFormat();
    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(new Configuration(), Collections.singletonList(indexPath1), new Path("output"), "id", Collections.singletonList("field"));
    conf = lucene2SeqConf.serialize();

    jobContext = new JobContext(conf, new JobID());
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(conf, indexPath1);
  }

  @Test
  public void testGetSplits() throws IOException, InterruptedException {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    //generate 3 segments
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc1);
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc2);
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc3);

    List<LuceneSegmentInputSplit> splits = inputFormat.getSplits(jobContext);
    Assert.assertEquals(3, splits.size());
  }
}
