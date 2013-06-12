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

@Deprecated
public final class MinHashDriver extends AbstractJob {

  public static final String NUM_HASH_FUNCTIONS = "numHashFunctions";
  public static final String KEY_GROUPS = "keyGroups";
  public static final String HASH_TYPE = "hashType";
  public static final String MIN_CLUSTER_SIZE = "minClusterSize";
  public static final String MIN_VECTOR_SIZE = "minVectorSize";
  public static final String NUM_REDUCERS = "numReducers";
  public static final String DEBUG_OUTPUT = "debugOutput";
  public static final String VECTOR_DIMENSION_TO_HASH  = "vectorDimensionToHash";

  static final String HASH_DIMENSION_VALUE = "value";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MinHashDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();


    addOption(MIN_CLUSTER_SIZE, "mcs", "Minimum points inside a cluster", String.valueOf(10));
    addOption(MIN_VECTOR_SIZE, "mvs", "Minimum size of vector to be hashed", String.valueOf(5));
    addOption(VECTOR_DIMENSION_TO_HASH, "vdh", "Dimension of vector to hash. Available types: (value, index). "
        + "Defaults to 'value'", HASH_DIMENSION_VALUE);
    addOption(HASH_TYPE, "ht", "Type of hash function to use. Available types: (linear, polynomial, murmur) ",
        HashFactory.HashType.MURMUR.toString());
    addOption(NUM_HASH_FUNCTIONS, "nh", "Number of hash functions to be used", String.valueOf(10));
    addOption(KEY_GROUPS, "kg", "Number of key groups to be used", String.valueOf(2));
    addOption(NUM_REDUCERS, "nr", "The number of reduce tasks. Defaults to 2", String.valueOf(2));
    addFlag(DEBUG_OUTPUT, "debug", "Output the whole vectors for debugging");
    addOption(DefaultOptionCreator.overwriteOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), getOutputPath());
    }

    int minClusterSize = Integer.valueOf(getOption(MIN_CLUSTER_SIZE));
    int minVectorSize = Integer.valueOf(getOption(MIN_VECTOR_SIZE));
    String dimensionToHash = getOption(VECTOR_DIMENSION_TO_HASH);
    String hashType = getOption(HASH_TYPE);
    int numHashFunctions = Integer.valueOf(getOption(NUM_HASH_FUNCTIONS));
    int keyGroups = Integer.valueOf(getOption(KEY_GROUPS));
    int numReduceTasks = Integer.parseInt(getOption(NUM_REDUCERS));
    boolean debugOutput = hasOption(DEBUG_OUTPUT);

    try {
      HashFactory.HashType.valueOf(hashType);
    } catch (IllegalArgumentException e) {
      System.err.println("Unknown hashType: " + hashType);
      return -1;
    }

    Class<? extends Writable> outputClass = debugOutput ? VectorWritable.class : Text.class;
    Class<? extends OutputFormat> outputFormatClass =
        debugOutput ? SequenceFileOutputFormat.class : TextOutputFormat.class;

    Job minHash = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class, MinHashMapper.class,
            Text.class, outputClass, MinHashReducer.class, Text.class, VectorWritable.class, outputFormatClass);

    Configuration minHashConfiguration = minHash.getConfiguration();
    minHashConfiguration.setInt(MIN_CLUSTER_SIZE, minClusterSize);
    minHashConfiguration.setInt(MIN_VECTOR_SIZE, minVectorSize);
    minHashConfiguration.set(VECTOR_DIMENSION_TO_HASH, dimensionToHash);
    minHashConfiguration.set(HASH_TYPE, hashType);
    minHashConfiguration.setInt(NUM_HASH_FUNCTIONS, numHashFunctions);
    minHashConfiguration.setInt(KEY_GROUPS, keyGroups);
    minHashConfiguration.setBoolean(DEBUG_OUTPUT, debugOutput);
    minHash.setNumReduceTasks(numReduceTasks);

    boolean succeeded = minHash.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    return 0;
  }
}
