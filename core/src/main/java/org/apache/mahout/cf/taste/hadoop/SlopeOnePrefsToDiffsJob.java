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
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;

import java.io.IOException;

/**
 */
public final class SlopeOnePrefsToDiffsJob {
  private SlopeOnePrefsToDiffsJob() {
  }

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

    FileSystem fs = FileSystem.get(outputPathPath.toUri(), jobConf);
    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.setInputFormat(TextInputFormat.class);
    FileInputFormat.setInputPaths(jobConf, prefsFilePath);

    jobConf.setMapperClass(SlopeOnePrefsToDiffsMapper.class);
    jobConf.setMapOutputKeyClass(Text.class);
    jobConf.setMapOutputValueClass(ItemPrefWritable.class);

    jobConf.setReducerClass(SlopeOnePrefsToDiffsReducer.class);
    jobConf.setOutputKeyClass(ItemItemWritable.class);
    jobConf.setOutputValueClass(DoubleWritable.class);

    jobConf.setOutputFormat(SequenceFileOutputFormat.class);
    SequenceFileOutputFormat.setOutputCompressionType(jobConf, SequenceFile.CompressionType.RECORD);
    FileOutputFormat.setOutputPath(jobConf, outputPathPath);

    return jobConf;
  }

}