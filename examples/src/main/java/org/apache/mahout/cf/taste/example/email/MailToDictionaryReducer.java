package org.apache.mahout.cf.taste.example.email;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;

/**
 * Key: the string id
 * Value: the count
 * Out Key: the string id
 * Out Value: the sum of the counts
 *
 **/
public class MailToDictionaryReducer extends
        Reducer<Text, VarIntWritable, Text, VarIntWritable> {

  @Override
  protected void reduce(Text key, Iterable<VarIntWritable> values, Context context) throws IOException, InterruptedException {
    int sum = 0;
    for (VarIntWritable value : values) {
      sum += value.get();
    }
    context.write(new Text(key), new VarIntWritable(sum));
  }
}
