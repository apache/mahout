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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;

import java.io.IOException;

/**
 */
public final class SlopeOneDiffsToAveragesJob {

  public static void main(String[] args) throws IOException {
    String prefsFile = args[0];
    String outputPath = args[1];
    JobConf jobConf = buildJobConf(prefsFile, outputPath);
    JobClient.runJob(jobConf);
  }

  public static JobConf buildJobConf(String prefsFile,
                                     String outputPath) throws IOException {

    Path prefsFilePath = new Path(prefsFile);
    Path outputPathPath = new Path(outputPath);

    JobConf jobConf = new JobConf(SlopeOnePrefsToDiffsJob.class);

    FileSystem fs = FileSystem.get(jobConf);
    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.setInputFormat(SequenceFileInputFormat.class);
    FileInputFormat.setInputPaths(jobConf, prefsFilePath);

    jobConf.setMapperClass(IdentityMapper.class);
    jobConf.setMapOutputKeyClass(ItemItemWritable.class);
    jobConf.setMapOutputValueClass(FloatWritable.class);

    jobConf.setReducerClass(SlopeOneDiffsToAveragesReducer.class);
    jobConf.setOutputKeyClass(ItemItemWritable.class);
    jobConf.setOutputValueClass(FloatWritable.class);

    jobConf.setOutputFormat(TextOutputFormat.class);
    FileOutputFormat.setOutputPath(jobConf, outputPathPath);

    return jobConf;
  }

}