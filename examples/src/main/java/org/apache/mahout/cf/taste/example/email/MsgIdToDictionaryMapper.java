package org.apache.mahout.cf.taste.example.email;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * Assumes the input is in the format created by {@link org.apache.mahout.text.SequenceFilesFromMailArchives}
 */
public class MsgIdToDictionaryMapper extends
        Mapper<Text, Text, Text, VarIntWritable> {
  public enum Counters {
    NO_MESSAGE_ID
  }

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    //message id is in the key: /201008/AANLkTikvVnhNH+Y5AGEwqd2=u0CFv2mCm0ce6E6oBnj1@mail.gmail.com
    String keyStr = key.toString();
    int idx = keyStr.lastIndexOf("/");
    String msgId = null;
    if (idx != -1) {
      msgId = keyStr.substring(idx + 1);
      context.write(new Text(msgId), new VarIntWritable(1));
    } else {
      context.getCounter(Counters.NO_MESSAGE_ID).increment(1);
    }
  }
}
