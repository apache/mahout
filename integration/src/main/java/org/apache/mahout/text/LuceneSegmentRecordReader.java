package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.lucene.index.SegmentInfoPerCommit;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.IOContext;

import java.io.IOException;

/**
 * {@link RecordReader} implementation for Lucene segments. Each {@link InputSplit} contains a separate Lucene segment.
 * Emits records consisting of a {@link Text} document ID and a null key.
 */
public class LuceneSegmentRecordReader extends RecordReader<Text, NullWritable> {

  public static final boolean READ_ONLY = true;
  public static final int USE_TERM_INFOS = 1;

  private SegmentReader segmentReader;
  private IndexSearcher searcher;
  private Scorer scorer;

  private int nextDocId;
  private Text key = new Text();

  @Override
  public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
    LuceneSegmentInputSplit inputSplit = (LuceneSegmentInputSplit) split;

    Configuration configuration = context.getConfiguration();
    LuceneStorageConfiguration lucene2SeqConfiguration = new LuceneStorageConfiguration(configuration);

    SegmentInfoPerCommit segmentInfo = inputSplit.getSegment(configuration);
    segmentReader = new SegmentReader(segmentInfo, USE_TERM_INFOS, IOContext.READ);//nocommit: Should we use IOContext.READONCE?


    searcher = new IndexSearcher(segmentReader);
    Weight weight = lucene2SeqConfiguration.getQuery().createWeight(searcher);
    scorer = weight.scorer(segmentReader.getContext(), true, false, null);//nocommit
  }

  @Override
  public boolean nextKeyValue() throws IOException, InterruptedException {
    nextDocId = scorer.nextDoc();

    return nextDocId != Scorer.NO_MORE_DOCS;
  }

  @Override
  public Text getCurrentKey() throws IOException, InterruptedException {
    key.set(String.valueOf(nextDocId));
    return key;
  }

  @Override
  public NullWritable getCurrentValue() throws IOException, InterruptedException {
    return NullWritable.get();
  }

  @Override
  public float getProgress() throws IOException, InterruptedException {
    return 0;
  }

  @Override
  public void close() throws IOException {
    segmentReader.close();
    //searcher.close();
  }
}
