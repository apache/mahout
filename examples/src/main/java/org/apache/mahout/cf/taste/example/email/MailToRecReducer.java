package org.apache.mahout.cf.taste.example.email;


import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 *
 *
 **/
public class MailToRecReducer extends Reducer<Text, LongWritable, Text, NullWritable>{
  //if true, then output weight
  private boolean useCounts = true;
  /**
   * We can either ignore how many times the user interacted (boolean) or output the number of times they interacted.
   */
  public static final String USE_COUNTS_PREFERENCE = "useBooleanPreferences";

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    useCounts = context.getConfiguration().getBoolean(USE_COUNTS_PREFERENCE, true);
  }

  @Override
  protected void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
    if (useCounts == false){
      context.write(new Text(key.toString()), null);
    } else {
      long sum = 0;
      for (LongWritable value : values) {
        sum++;
      }
      context.write(new Text(key.toString() + "," + sum), null);
    }
  }
}
