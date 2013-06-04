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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;

public final class MinHashDriver extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MinHashDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(MinhashOptionCreator.minClusterSizeOption().create());
    addOption(MinhashOptionCreator.minVectorSizeOption().create());
    addOption(MinhashOptionCreator.vectorDimensionToHashOption().create());
    addOption(MinhashOptionCreator.hashTypeOption().create());
    addOption(MinhashOptionCreator.numHashFunctionsOption().create());
    addOption(MinhashOptionCreator.keyGroupsOption().create());
    addOption(MinhashOptionCreator.numReducersOption().create());
    addOption(MinhashOptionCreator.debugOutputOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), getOutputPath());
    }

    int minClusterSize = Integer.valueOf(getOption(MinhashOptionCreator.MIN_CLUSTER_SIZE));
    int minVectorSize = Integer.valueOf(getOption(MinhashOptionCreator.MIN_VECTOR_SIZE));
    String dimensionToHash = getOption(MinhashOptionCreator.VECTOR_DIMENSION_TO_HASH);
    String hashType = getOption(MinhashOptionCreator.HASH_TYPE);
    int numHashFunctions = Integer.valueOf(getOption(MinhashOptionCreator.NUM_HASH_FUNCTIONS));
    int keyGroups = Integer.valueOf(getOption(MinhashOptionCreator.KEY_GROUPS));
    int numReduceTasks = Integer.parseInt(getOption(MinhashOptionCreator.NUM_REDUCERS));
    boolean debugOutput = hasOption(MinhashOptionCreator.DEBUG_OUTPUT);

    Class<? extends Writable> outputClass = debugOutput ? VectorWritable.class : Text.class;
    Class<? extends OutputFormat> outputFormatClass =
        debugOutput ? SequenceFileOutputFormat.class : TextOutputFormat.class;

    Job minHash = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class, MinHashMapper.class,
            Text.class, outputClass, MinHashReducer.class, Text.class, VectorWritable.class, outputFormatClass);

    Configuration minHashConfiguration = minHash.getConfiguration();
    minHashConfiguration.setInt(MinhashOptionCreator.MIN_CLUSTER_SIZE, minClusterSize);
    minHashConfiguration.setInt(MinhashOptionCreator.MIN_VECTOR_SIZE, minVectorSize);
    minHashConfiguration.set(MinhashOptionCreator.VECTOR_DIMENSION_TO_HASH, dimensionToHash);
    minHashConfiguration.set(MinhashOptionCreator.HASH_TYPE, hashType);
    minHashConfiguration.setInt(MinhashOptionCreator.NUM_HASH_FUNCTIONS, numHashFunctions);
    minHashConfiguration.setInt(MinhashOptionCreator.KEY_GROUPS, keyGroups);
    minHashConfiguration.setBoolean(MinhashOptionCreator.DEBUG_OUTPUT, debugOutput);
    minHash.setNumReduceTasks(numReduceTasks);

    boolean succeeded = minHash.waitForCompletion(true);
    if (!succeeded) {
     return -1;
    }

    return 0;
  }
}
