package org.apache.mahout.classifier.bayes;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Progressable;

/**
 * This class extends the MultipleOutputFormat, allowing to write the output
 * data to different output files in Text output format.
 */
public class MultipleTextOutputFormat<K, V>
    extends MultipleOutputFormat<K, V> {

  private TextOutputFormat<K, V> theTextOutputFormat = null;

  @Override
  protected RecordWriter<K, V> getBaseRecordWriter(FileSystem fs, Configuration conf, String name, Progressable arg3)
      throws IOException {
    if (theTextOutputFormat == null) {
      theTextOutputFormat = new TextOutputFormat<K, V>();
    }
    try {
      return theTextOutputFormat.getRecordWriter(new TaskAttemptContext(conf, new TaskAttemptID()));
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return null;
  }

  @Override
  public RecordWriter<K, V> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
    if (theTextOutputFormat == null) {
      theTextOutputFormat = new TextOutputFormat<K, V>();
    }
    return theTextOutputFormat.getRecordWriter(job);
  }
}
