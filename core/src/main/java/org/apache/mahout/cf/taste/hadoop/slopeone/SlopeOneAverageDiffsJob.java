/*
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
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarLongWritable;

public final class SlopeOneAverageDiffsJob extends AbstractJob {
  
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
    addInputOption();
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    
    Path prefsFile = getInputPath();
    Path outputPath = getOutputPath();
    Path averagesOutputPath = new Path(parsedArgs.get("--tempDir"));

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job prefsToDiffsJob = prepareJob(prefsFile,
                                       averagesOutputPath,
                                       TextInputFormat.class,
                                       ToItemPrefsMapper.class,
                                       VarLongWritable.class,
                                       EntityPrefWritable.class,
                                       SlopeOnePrefsToDiffsReducer.class,
                                       EntityEntityWritable.class,
                                       FloatWritable.class,
                                       SequenceFileOutputFormat.class);
      prefsToDiffsJob.waitForCompletion(true);
    }


    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job diffsToAveragesJob = prepareJob(averagesOutputPath,
                                          outputPath,
                                          SequenceFileInputFormat.class,
                                          Mapper.class,
                                          EntityEntityWritable.class,
                                          FloatWritable.class,
                                          SlopeOneDiffsToAveragesReducer.class,
                                          EntityEntityWritable.class,
                                          FullRunningAverageAndStdDevWritable.class,
                                          TextOutputFormat.class);
      FileOutputFormat.setOutputCompressorClass(diffsToAveragesJob, GzipCodec.class);
      diffsToAveragesJob.waitForCompletion(true);
    }
    return 0;
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SlopeOneAverageDiffsJob(), args);
  }
  
}