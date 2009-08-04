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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.StringUtils;

import java.io.IOException;

public final class SlopeOnePrefsToDiffsJob extends Job {

  private SlopeOnePrefsToDiffsJob(Configuration jobConf) throws IOException {
    super(jobConf);
  }

  public static void main(String[] args) throws Exception {
    String prefsFile = args[0];
    String outputPath = args[1];
    Configuration jobConf = buildJobConf(prefsFile, outputPath);
    Job job = new SlopeOnePrefsToDiffsJob(jobConf);
    job.waitForCompletion(true);
  }

  public static Configuration buildJobConf(String prefsFile,
                                           String outputPath) throws IOException {

    Configuration jobConf = new Configuration();
    FileSystem fs = FileSystem.get(jobConf);

    Path prefsFilePath = new Path(prefsFile).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.setClass("mapred.input.format.class", TextInputFormat.class, InputFormat.class);
    jobConf.set("mapred.input.dir", StringUtils.escapeString(prefsFilePath.toString()));

    jobConf.setClass("mapred.mapper.class", SlopeOnePrefsToDiffsMapper.class, Mapper.class);
    jobConf.setClass("mapred.mapoutput.key.class", Text.class, Object.class);
    jobConf.setClass("mapred.mapoutput.value.class", ItemPrefWritable.class, Object.class);

    jobConf.setClass("mapred.reducer.class", SlopeOnePrefsToDiffsReducer.class, Reducer.class);
    jobConf.setClass("mapred.output.key.class", ItemItemWritable.class, Object.class);
    jobConf.setClass("mapred.output.value.class", FloatWritable.class, Object.class);

    jobConf.setClass("mapred.output.format.class", SequenceFileOutputFormat.class, OutputFormat.class);
    jobConf.set("mapred.output.dir", StringUtils.escapeString(outputPathPath.toString()));

    return jobConf;
  }

}