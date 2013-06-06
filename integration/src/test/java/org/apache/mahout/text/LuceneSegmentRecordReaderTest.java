package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.lucene.index.*;
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
import static org.junit.Assert.assertEquals;

public class LuceneSegmentRecordReaderTest extends AbstractLuceneStorageTest {

  private LuceneSegmentRecordReader recordReader;
  private Configuration configuration;

  @Before
  public void before() throws IOException, InterruptedException {
    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(new Configuration(), asList(getIndexPath()), new Path("output"), "id", asList("field"));
    configuration = lucene2SeqConf.serialize();

    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    commitDocuments(doc1, doc2, doc3);

    SegmentInfos segmentInfos = new SegmentInfos();
    segmentInfos.read(getDirectory());

    SegmentInfo segmentInfo = segmentInfos.asList().get(0);
    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(getIndexPath(), segmentInfo.name, segmentInfo.sizeInBytes(true));

    TaskAttemptContext context = new TaskAttemptContext(configuration, new TaskAttemptID());

    recordReader = new LuceneSegmentRecordReader();
    recordReader.initialize(inputSplit, context);
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(configuration, getIndexPath());
  }

  @Test
  public void testKey() throws Exception {
    recordReader.nextKeyValue();
    assertEquals("0", recordReader.getCurrentKey().toString());
    recordReader.nextKeyValue();
    assertEquals("1", recordReader.getCurrentKey().toString());
    recordReader.nextKeyValue();
    assertEquals("2", recordReader.getCurrentKey().toString());
  }

  @Test
  public void testGetCurrentValue() throws Exception {
    assertEquals(NullWritable.get(), recordReader.getCurrentValue());
  }
}
