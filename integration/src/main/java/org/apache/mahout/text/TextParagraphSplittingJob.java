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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;

import java.io.IOException;

public class TextParagraphSplittingJob extends AbstractJob {

  @Override
  public int run(String[] strings) throws Exception {
    Configuration originalConf = getConf();
    Job job = prepareJob(new Path(originalConf.get("mapred.input.dir")),
                         new Path(originalConf.get("mapred.output.dir")),
                         SequenceFileInputFormat.class,
                         SplitMap.class,
                         Text.class,
                         Text.class,
                         Reducer.class,
                         Text.class,
                         Text.class,
                         SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    boolean succeeded = job.waitForCompletion(true);
    return succeeded ? 0 : -1;
  }

  public static class SplitMap extends Mapper<Text,Text,Text,Text> {

    @Override
    protected void map(Text key, Text text, Context context) throws IOException, InterruptedException {
      Text outText = new Text();
      int loc = 0;
      while (loc >= 0 && loc < text.getLength()) {
        int nextLoc = text.find("\n\n", loc + 1);
        if (nextLoc > 0) {
          outText.set(text.getBytes(), loc, nextLoc - loc);
          context.write(key, outText);
        }
        loc = nextLoc;
      }
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TextParagraphSplittingJob(), args);
  }
}
