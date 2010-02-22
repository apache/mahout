package org.apache.mahout.text;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;

import java.io.IOException;
import java.util.Map;


public class TextParagraphSplittingJob extends AbstractJob {

  @Override
  public int run(String[] strings) throws Exception {
    Map<String,String> args = parseArguments(strings);
    JobConf conf = prepareJobConf(args.get("--input"),
                                  args.get("--output"),
                                  args.get("--jarFile"),
                                  SequenceFileInputFormat.class,
                                  SplitMap.class,
                                  Text.class,
                                  Text.class,
                                  Reducer.class,
                                  Text.class,
                                  Text.class,
                                  SequenceFileOutputFormat.class);
    conf.setNumReduceTasks(0);

    JobClient.runJob(conf).waitForCompletion();
    return 1;
  }

  public static class SplitMap extends MapReduceBase implements Mapper<Text,Text,Text,Text> {

    @Override
    public void map(Text key,
                    Text text,
                    OutputCollector<Text, Text> out,
                    Reporter reporter) throws IOException {
      Text outText = new Text();
      int loc = 0;
      while(loc >= 0 && loc < text.getLength()) {
        int nextLoc = text.find("\n\n", loc+1);
        if(nextLoc > 0) {
          outText.set(text.getBytes(), loc, (nextLoc - loc));
          out.collect(key, outText);
        }
        loc = nextLoc;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TextParagraphSplittingJob(), args);
  }
}
