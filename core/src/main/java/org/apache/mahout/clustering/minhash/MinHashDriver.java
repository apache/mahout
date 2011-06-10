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

package org.apache.mahout.clustering.minhash;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.MinhashOptionCreator;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public final class MinHashDriver extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new MinHashDriver(), args);
  }

  private void runJob(Path input, 
                      Path output,
                      int minClusterSize,
                      int minVectorSize, 
                      String hashType, 
                      int numHashFunctions, 
                      int keyGroups,
                      int numReduceTasks, 
                      boolean debugOutput) throws IOException, ClassNotFoundException, InterruptedException {
    Configuration conf = getConf();

    conf.setInt(MinhashOptionCreator.MIN_CLUSTER_SIZE, minClusterSize);
    conf.setInt(MinhashOptionCreator.MIN_VECTOR_SIZE, minVectorSize);
    conf.set(MinhashOptionCreator.HASH_TYPE, hashType);
    conf.setInt(MinhashOptionCreator.NUM_HASH_FUNCTIONS, numHashFunctions);
    conf.setInt(MinhashOptionCreator.KEY_GROUPS, keyGroups);
    conf.setBoolean(MinhashOptionCreator.DEBUG_OUTPUT, debugOutput);

    Class<? extends Writable> outputClass = debugOutput ? VectorWritable.class : Text.class;
    Class<? extends OutputFormat> outputFormatClass =
        debugOutput ? SequenceFileOutputFormat.class : TextOutputFormat.class;
    
    Job job = new Job(conf, "MinHash Clustering");
    job.setJarByClass(MinHashDriver.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(MinHashMapper.class);
    job.setReducerClass(MinHashReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(outputFormatClass);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(outputClass);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(outputClass);

    job.setNumReduceTasks(numReduceTasks);

    job.waitForCompletion(true);
  }

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(MinhashOptionCreator.minClusterSizeOption().create());
    addOption(MinhashOptionCreator.minVectorSizeOption().create());
    addOption(MinhashOptionCreator.hashTypeOption().create());
    addOption(MinhashOptionCreator.numHashFunctionsOption().create());
    addOption(MinhashOptionCreator.keyGroupsOption().create());
    addOption(MinhashOptionCreator.numReducersOption().create());
    addOption(MinhashOptionCreator.debugOutputOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    int minClusterSize = Integer.valueOf(getOption(MinhashOptionCreator.MIN_CLUSTER_SIZE));
    int minVectorSize = Integer.valueOf(getOption(MinhashOptionCreator.MIN_VECTOR_SIZE));
    String hashType = getOption(MinhashOptionCreator.HASH_TYPE);
    int numHashFunctions = Integer.valueOf(getOption(MinhashOptionCreator.NUM_HASH_FUNCTIONS));
    int keyGroups = Integer.valueOf(getOption(MinhashOptionCreator.KEY_GROUPS));
    int numReduceTasks = Integer.parseInt(getOption(MinhashOptionCreator.NUM_REDUCERS));
    boolean debugOutput = Boolean.parseBoolean(getOption(MinhashOptionCreator.DEBUG_OUTPUT));

    runJob(input,
           output,
           minClusterSize,
           minVectorSize,
           hashType,
           numHashFunctions,
           keyGroups,
           numReduceTasks,
           debugOutput);
    return 0;
  }
}
