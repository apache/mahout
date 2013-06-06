package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.lucene.index.SegmentInfoPerCommit;
import org.apache.lucene.index.SegmentInfos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link InputFormat} implementation which splits a Lucene index at the segment level.
 */
public class LuceneSegmentInputFormat extends InputFormat {

  private static final Logger LOG = LoggerFactory.getLogger(LuceneSegmentInputFormat.class);

  @Override
  public List<LuceneSegmentInputSplit> getSplits(JobContext context) throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();

    LuceneStorageConfiguration lucene2SeqConfiguration = new LuceneStorageConfiguration(configuration);

    List<LuceneSegmentInputSplit> inputSplits = new ArrayList<LuceneSegmentInputSplit>();

    List<Path> indexPaths = lucene2SeqConfiguration.getIndexPaths();
    for (Path indexPath : indexPaths) {
      ReadOnlyFileSystemDirectory directory = new ReadOnlyFileSystemDirectory(FileSystem.get(configuration), indexPath, false, configuration);
      SegmentInfos segmentInfos = new SegmentInfos();
      segmentInfos.read(directory);

      for (SegmentInfoPerCommit segmentInfo : segmentInfos) {
        LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath, segmentInfo.info.name, segmentInfo.sizeInBytes());
        inputSplits.add(inputSplit);
        LOG.info("Created {} byte input split for index '{}' segment {}", new Object[]{segmentInfo.sizeInBytes(), indexPath.toUri(), segmentInfo.info.name});
      }
    }

    return inputSplits;
  }

  @Override
  public RecordReader<Text, NullWritable> createRecordReader(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
    LuceneSegmentRecordReader luceneSegmentRecordReader = new LuceneSegmentRecordReader();
    luceneSegmentRecordReader.initialize(inputSplit, context);
    return luceneSegmentRecordReader;
  }
}
