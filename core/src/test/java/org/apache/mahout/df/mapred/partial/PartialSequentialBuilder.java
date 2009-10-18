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

package org.apache.mahout.df.mapred.partial;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.mapreduce.partial.InterResults;
import org.apache.mahout.df.mapreduce.partial.TreeID;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simulates the Partial mapreduce implementation in a sequential manner. Must
 * receive a seed
 */
public class PartialSequentialBuilder extends PartialBuilder {

  private static final Logger log = LoggerFactory.getLogger(PartialSequentialBuilder.class);

  protected PartialOutputCollector firstOutput;

  protected PartialOutputCollector secondOutput;

  protected final Dataset dataset;

  /** first instance id in hadoop's order */
  protected int[] firstIds;
  
  /** partitions' sizes in hadoop order */
  protected int[] sizes;

  public PartialSequentialBuilder(TreeBuilder treeBuilder, Path dataPath,
      Dataset dataset, long seed, Configuration conf) {
    super(treeBuilder, dataPath, new Path("notUsed"), seed, conf);
    this.dataset = dataset;
  }

  public PartialSequentialBuilder(TreeBuilder treeBuilder, Path dataPath,
      Dataset dataset, long seed) {
    this(treeBuilder, dataPath, dataset, seed, new Configuration());
  }

  @Override
  protected void configureJob(JobConf job, int nbTrees, boolean oobEstimate)
      throws IOException {
    
    int numMaps = job.getNumMapTasks();

    super.configureJob(job, nbTrees, oobEstimate);

    // PartialBuilder sets the number of maps to 1 if we are running in 'local'
    job.setNumMapTasks(numMaps);
  }

  @Override
  protected void runJob(JobConf job) throws IOException {
    // retrieve the splits
    TextInputFormat input = (TextInputFormat) job.getInputFormat();
    InputSplit[] splits = input.getSplits(job, job.getNumMapTasks());
    log.debug("Nb splits : " + splits.length);

    InputSplit[] sorted = Arrays.copyOf(splits, splits.length);
    Builder.sortSplits(sorted);

    int numTrees = Builder.getNbTrees(job); // total number of trees

    firstOutput = new PartialOutputCollector(numTrees);
    Reporter reporter = Reporter.NULL;

    firstIds = new int[splits.length];
    sizes = new int[splits.length];
    
    // to compute firstIds, process the splits in file order
    int firstId = 0;
    long slowest = 0; // duration of slowest map
    for (InputSplit split : splits) {
      int hp = ArrayUtils.indexOf(sorted, split); // hadoop's partition

      RecordReader<LongWritable, Text> reader = input.getRecordReader(split, job, reporter);

      LongWritable key = reader.createKey();
      Text value = reader.createValue();

      Step1Mapper mapper = new MockStep1Mapper(treeBuilder, dataset, seed,
          hp, splits.length, numTrees);

      long time = System.currentTimeMillis();

      firstIds[hp] = firstId;

      while (reader.next(key, value)) {
        mapper.map(key, value, firstOutput, reporter);
        firstId++;
        sizes[hp]++;
      }

      mapper.close();

      time = System.currentTimeMillis() - time;
      log.info("Duration : " + DFUtils.elapsedTime(time));

      if (time > slowest) {
        slowest = time;
      }
    }

    log.info("Longest duration : " + DFUtils.elapsedTime(slowest));
  }

  @Override
  protected DecisionForest parseOutput(JobConf job, PredictionCallback callback)
      throws IOException {
    DecisionForest forest = processOutput(firstOutput.keys, firstOutput.values, callback);

    if (isStep2(job)) {
      Path forestPath = new Path(getOutputPath(job), "step1.inter");
      FileSystem fs = forestPath.getFileSystem(job);
      
      Node[] trees = new Node[forest.getTrees().size()];
      forest.getTrees().toArray(trees);
      InterResults.store(fs, forestPath, firstOutput.keys, trees, sizes);

      log.info("***********");
      log.info("Second Step");
      log.info("***********");
      secondStep(job, forestPath, callback);

      processOutput(secondOutput.keys, secondOutput.values, callback);
    }

    return forest;
  }

  /**
   * extract the decision forest and call the callback after correcting the instance ids
   * 
   * @param keys
   * @param values
   * @param callback
   * @return
   */
  protected DecisionForest processOutput(TreeID[] keys, MapredOutput[] values, PredictionCallback callback) {
    List<Node> trees = new ArrayList<Node>();

    for (int index = 0; index < keys.length; index++) {
      TreeID key = keys[index];
      MapredOutput value = values[index];

      trees.add(value.getTree());

      int[] predictions = value.getPredictions();
      for (int id = 0; id < predictions.length; id++) {
        callback.prediction(key.treeId(), firstIds[key.partition()] + id,
            predictions[id]);
      }
    }
    
    return new DecisionForest(trees);
  }

  /**
   * The second step uses the trees to predict the rest of the instances outside
   * their own partition
   * 
   * @throws IOException
   * 
   */
  protected void secondStep(JobConf job, Path forestPath,
      PredictionCallback callback) throws IOException {
    // retrieve the splits
    TextInputFormat input = (TextInputFormat) job.getInputFormat();
    InputSplit[] splits = input.getSplits(job, job.getNumMapTasks());
    log.debug("Nb splits : " + splits.length);

    Builder.sortSplits(splits);

    int numTrees = Builder.getNbTrees(job); // total number of trees

    // compute the expected number of outputs
    int total = 0;
    for (int p = 0; p < splits.length; p++) {
      total += Step2Mapper.nbConcerned(splits.length, numTrees, p);
    }

    secondOutput = new PartialOutputCollector(total);
    Reporter reporter = Reporter.NULL;
    long slowest = 0; // duration of slowest map

    for (int partition = 0; partition < splits.length; partition++) {
      InputSplit split = splits[partition];
      RecordReader<LongWritable, Text> reader = input.getRecordReader(split,
          job, reporter);

      LongWritable key = reader.createKey();
      Text value = reader.createValue();

      // load the output of the 1st step
      int nbConcerned = Step2Mapper.nbConcerned(splits.length, numTrees,
          partition);
      TreeID[] fsKeys = new TreeID[nbConcerned];
      Node[] fsTrees = new Node[nbConcerned];

      FileSystem fs = forestPath.getFileSystem(job);
      int numInstances = InterResults.load(fs, forestPath, splits.length,
          numTrees, partition, fsKeys, fsTrees);

      Step2Mapper mapper = new Step2Mapper();
      mapper.configure(partition, dataset, fsKeys, fsTrees, numInstances);

      long time = System.currentTimeMillis();

      while (reader.next(key, value)) {
        mapper.map(key, value, secondOutput, reporter);
      }

      mapper.close();

      time = System.currentTimeMillis() - time;
      log.info("Duration : " + DFUtils.elapsedTime(time));

      if (time > slowest) {
        slowest = time;
      }
    }

    log.info("Longest duration : " + DFUtils.elapsedTime(slowest));
  }

  /**
   * Special Step1Mapper that can be configured without using a Configuration
   * 
   */
  protected static class MockStep1Mapper extends Step1Mapper {
    protected MockStep1Mapper(TreeBuilder treeBuilder, Dataset dataset, Long seed,
        int partition, int numMapTasks, int numTrees) {
      configure(false, true, treeBuilder, dataset);
      configure(seed, partition, numMapTasks, numTrees);
    }

  }

}
