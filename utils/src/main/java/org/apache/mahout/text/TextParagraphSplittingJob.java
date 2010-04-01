/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
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
import org.apache.mahout.common.AbstractJob;

import java.io.IOException;

public class TextParagraphSplittingJob extends AbstractJob {

  @Override
  public int run(String[] strings) throws Exception {
    Configuration originalConf = getConf();
    JobConf conf = prepareJobConf(originalConf.get("mapred.input.dir"),
                                  originalConf.get("mapred.output.dir"),
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
