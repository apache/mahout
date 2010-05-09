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

package org.apache.mahout.cf.taste.hadoop.slopeone;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;

public final class SlopeOneAverageDiffsJob extends AbstractJob {
  
  @Override
  public int run(String[] args) throws IOException {
    
    Map<String,String> parsedArgs = AbstractJob.parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    
    Configuration originalConf = getConf();
    String prefsFile = originalConf.get("mapred.input.dir");
    String outputPath = originalConf.get("mapred.output.dir");
    String averagesOutputPath = parsedArgs.get("--tempDir");
    
    JobConf prefsToDiffsJobConf = prepareJobConf(prefsFile, averagesOutputPath,
      TextInputFormat.class, ToItemPrefsMapper.class, VLongWritable.class, EntityPrefWritable.class,
      SlopeOnePrefsToDiffsReducer.class, EntityEntityWritable.class, FloatWritable.class,
      SequenceFileOutputFormat.class);
    JobClient.runJob(prefsToDiffsJobConf);
    
    JobConf diffsToAveragesJobConf = prepareJobConf(averagesOutputPath, outputPath,
      SequenceFileInputFormat.class, IdentityMapper.class, EntityEntityWritable.class, FloatWritable.class,
      SlopeOneDiffsToAveragesReducer.class, EntityEntityWritable.class, FloatWritable.class,
      TextOutputFormat.class);
    diffsToAveragesJobConf.setClass("mapred.output.compression.codec", GzipCodec.class,
      CompressionCodec.class);
    JobClient.runJob(diffsToAveragesJobConf);
    return 0;
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SlopeOneAverageDiffsJob(), args);
  }
  
}