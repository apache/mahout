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

package org.apache.mahout.fpm.pfpgrowth.dataset;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;

public final class KeyBasedStringTupleGrouper {
  
  private KeyBasedStringTupleGrouper() { }
  
  public static void startJob(Parameters params) throws IOException,
                                                InterruptedException,
                                                ClassNotFoundException {
    Configuration conf = new Configuration();
    
    conf.set("job.parameters", params.toString());
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    conf.set("mapred.map.output.compression.codec", "org.apache.hadoop.io.compress.GzipCodec");
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    String input = params.get("input");
    Job job = new Job(conf, "Generating dataset based from input" + input);
    job.setJarByClass(KeyBasedStringTupleGrouper.class);
    
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(StringTuple.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    
    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get("output"));
    FileOutputFormat.setOutputPath(job, outPath);
    
    HadoopUtil.delete(conf, outPath);

    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(KeyBasedStringTupleMapper.class);
    job.setCombinerClass(KeyBasedStringTupleCombiner.class);
    job.setReducerClass(KeyBasedStringTupleReducer.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
}
