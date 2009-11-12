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

package org.apache.mahout.df.mapreduce.partial;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.partial.Step0Job.Step0Output;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds a random forest using partial data. Each mapper uses only the data
 * given by its InputSplit
 */
public class PartialBuilder extends Builder {

  private static final Logger log = LoggerFactory.getLogger(PartialBuilder.class);

  /**
   * Indicates if we should run the second step of the builder.<br>
   * This parameter is only meant for debuging, so we keep it protected.
   * 
   * @param conf
   * @return
   */
  protected static boolean isStep2(Configuration conf) {
    return conf.getBoolean("debug.mahout.rf.partial.step2", true);
  }

  /**
   * Should run the second step of the builder ?
   * 
   * @param conf
   * @param value true to indicate that the second step will be launched
   * 
   */
  protected static void setStep2(Configuration conf, boolean value) {
    conf.setBoolean("debug.mahout.rf.partial.step2", value);
  }

  public PartialBuilder(TreeBuilder treeBuilder, Path dataPath,
      Path datasetPath, Long seed) {
    this(treeBuilder, dataPath, datasetPath, seed, new Configuration());
  }

  public PartialBuilder(TreeBuilder treeBuilder, Path dataPath,
      Path datasetPath, Long seed, Configuration conf) {
    super(treeBuilder, dataPath, datasetPath, seed, conf);
  }

  
  @Override
  protected void configureJob(Job job, int nbTrees, boolean oobEstimate)
      throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));

    job.setOutputKeyClass(TreeID.class);
    job.setOutputValueClass(MapredOutput.class);

    job.setMapperClass(Step1Mapper.class);
    job.setNumReduceTasks(0); // no reducers

    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }

  @Override
  protected DecisionForest parseOutput(Job job, PredictionCallback callback)
      throws IOException, ClassNotFoundException, InterruptedException {
    Configuration conf = job.getConfiguration();
    
    int numTrees = getNbTrees(conf);

    Path outputPath = getOutputPath(conf);

    log.info("Computing partitions' first ids...");
    Step0Job step0 = new Step0Job(getOutputPath(conf), getDataPath(), getDatasetPath());
    Step0Output[] partitions = step0.run(new Configuration(conf));

    log.info("Processing the output...");
    TreeID[] keys = new TreeID[numTrees];
    Node[] trees = new Node[numTrees];
    int[] firstIds = Step0Output.extractFirstIds(partitions);
    processOutput(job, outputPath, firstIds, keys, trees, callback);

    // JobClient should have updated numMaps to the correct number of maps
    int numMaps = partitions.length;

    // call the second step in order to complete the oob predictions
    if (callback != null && numMaps > 1 && isStep2(conf)) {
      log.info("*****************************");
      log.info("Second Step");
      log.info("*****************************");
      Step2Job step2 = new Step2Job(getOutputPath(conf), getDataPath(), getDatasetPath(), partitions);

      step2.run(new Configuration(conf), keys, trees, callback);
    }

    return new DecisionForest(Arrays.asList(trees));
  }

  /**
   * Processes the output from the output path.<br>
   * 
   * @param job
   * @param outputPath directory that contains the output of the job
   * @param firstIds partitions' first ids in hadoop's order
   * @param keys
   * @param callback can be null
   * @throws IOException
   */
  protected static void processOutput(Job job, Path outputPath,
      int[] firstIds, TreeID[] keys, Node[] trees, PredictionCallback callback)
      throws IOException {
    if (keys.length != trees.length) {
      throw new IllegalArgumentException("keys.length != trees.length");
    }
    
    Configuration conf = job.getConfiguration();
    
    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);

    // read all the outputs
    TreeID key = new TreeID();
    MapredOutput value = new MapredOutput();
    
    int index = 0;
    for (Path path : outfiles) {
      Reader reader = new Reader(fs, path, conf);

      try {
        while (reader.next(key, value)) {
          if (keys != null) {
            keys[index] = key.clone();
          }
          
          if (trees != null) {
            trees[index] = value.getTree();
          }
          
          processOutput(firstIds, key, value, callback);
          
          index++;
        }
      } finally {
        reader.close();
      }
    }

    // make sure we got all the keys/values
    if (index != keys.length) {
      throw new IllegalStateException();
    }
  }

  /**
   * Process the output, extracting the trees and passing the predictions to the
   * callback
   * 
   * @param firstIds partitions' first ids in hadoop's order
   * @param callback
   * @return
   */
  private static void processOutput(int[] firstIds, TreeID key,
      MapredOutput value, PredictionCallback callback) {

    if (callback != null) {
      int[] predictions = value.getPredictions();

      for (int instanceId = 0; instanceId < predictions.length; instanceId++) {
        callback.prediction(key.treeId(), firstIds[key.partition()] + instanceId, 
            predictions[instanceId]);
      }
    }
  }
}
