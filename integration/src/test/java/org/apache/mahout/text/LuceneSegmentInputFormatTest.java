package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
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

public class LuceneSegmentInputFormatTest {

  private LuceneSegmentInputFormat inputFormat;
  private JobContext jobContext;
  private Path indexPath;
  private Configuration conf;
  private FSDirectory directory;

  @Before
  public void before() throws IOException {
    inputFormat = new LuceneSegmentInputFormat();
    indexPath = new Path("index");

    LuceneStorageConfiguration lucene2SeqConf = new LuceneStorageConfiguration(new Configuration(), asList(indexPath), new Path("output"), "id", asList("field"));
    conf = lucene2SeqConf.serialize();

    jobContext = new JobContext(conf, new JobID());
    directory = FSDirectory.open(new File(indexPath.toString()));
  }
  
  @After
  public void after() throws IOException {
    HadoopUtil.delete(conf, indexPath);
  }

  @Test
  public void testGetSplits() throws IOException, InterruptedException {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");
    List<SingleFieldDocument> documents = asList(doc1, doc2, doc3);

    for (SingleFieldDocument singleFieldDocument : documents) {
      commitDocument(singleFieldDocument);
    }

    List<LuceneSegmentInputSplit> splits = inputFormat.getSplits(jobContext);
    assertEquals(3, splits.size());
  }

  private void commitDocument(SingleFieldDocument doc) throws IOException {
    IndexWriterConfig conf = new IndexWriterConfig(Version.LUCENE_35, new DefaultAnalyzer());
    IndexWriter indexWriter = new IndexWriter(directory, conf);
    indexWriter.addDocument(doc.asLuceneDocument());
    indexWriter.commit();
    indexWriter.close();
  }
}
