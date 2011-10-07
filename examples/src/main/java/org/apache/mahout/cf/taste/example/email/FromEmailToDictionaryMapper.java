package org.apache.mahout.cf.taste.example.email;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;

/**
 *  Assumes the input is in the format created by {@link org.apache.mahout.text.SequenceFilesFromMailArchives}
 *
 **/
public class FromEmailToDictionaryMapper extends
        Mapper<Text, Text, Text, VarIntWritable> {
  private String separator = "\n";


  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    separator = context.getConfiguration().get(EmailUtility.SEPARATOR);
  }

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    //From is in the value
    String valStr = value.toString();
    int idx = valStr.indexOf(separator);
    if (idx != -1){
      String full = valStr.substring(0, idx);
      //do some cleanup to normalize some things, like: Key: karthik ananth <karthik.jcecs@gmail.com>: Value: 178
            //Key: karthik ananth [mailto:karthik.jcecs@gmail.com]=20: Value: 179
      //TODO: is there more to clean up here?
      full = EmailUtility.cleanUpEmailAddress(full);

      context.write(new Text(full), new VarIntWritable(1));
    }

  }
}
