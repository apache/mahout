package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import java.io.IOException;

/**
 * Generates a sequence file from a Lucene index via MapReduce. Uses a specified id field as the key and a content field as the value.
 * Configure this class with a {@link LuceneStorageConfiguration} bean.
 */
public class SequenceFilesFromLuceneStorageMRJob {

  public void run(LuceneStorageConfiguration lucene2seqConf) {
    try {
      Configuration configuration = lucene2seqConf.serialize();

      Job job = new Job(configuration, "LuceneIndexToSequenceFiles: " + lucene2seqConf.getIndexPaths() + " -> M/R -> " + lucene2seqConf.getSequenceFilesOutputPath());

      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(Text.class);

      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);

      job.setOutputFormatClass(SequenceFileOutputFormat.class);

      job.setMapperClass(SequenceFilesFromLuceneStorageMapper.class);

      job.setInputFormatClass(LuceneSegmentInputFormat.class);

      for (Path indexPath : lucene2seqConf.getIndexPaths()) {
        FileInputFormat.addInputPath(job, indexPath);
      }

      FileOutputFormat.setOutputPath(job, lucene2seqConf.getSequenceFilesOutputPath());

      job.setJarByClass(SequenceFilesFromLuceneStorageMRJob.class);
      job.setNumReduceTasks(0);

      job.waitForCompletion(true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
}
