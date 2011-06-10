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
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simulates the Partial mapreduce implementation in a sequential manner. Must
 * receive a seed
 */
public class PartialSequentialBuilder extends PartialBuilder {

  private static final Logger log = LoggerFactory.getLogger(PartialSequentialBuilder.class);

  private MockContext firstOutput;

  private MockContext secondOutput;

  private final Dataset dataset;

  /** first instance id in hadoop's order */
  private int[] firstIds;
  
  /** partitions' sizes in hadoop order */
  private int[] sizes;

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
  protected void configureJob(Job job, int nbTrees, boolean oobEstimate)
      throws IOException {
    Configuration conf = job.getConfiguration();
    
    int num = conf.getInt("mapred.map.tasks", -1);

    super.configureJob(job, nbTrees, oobEstimate);

    // PartialBuilder sets the number of maps to 1 if we are running in 'local'
    conf.setInt("mapred.map.tasks", num);
  }

  @Override
  protected boolean runJob(Job job) throws IOException, InterruptedException {
    Configuration conf = job.getConfiguration();
    
    // retrieve the splits
    TextInputFormat input = new TextInputFormat();
    List<InputSplit> splits = input.getSplits(job);
    
    int nbSplits = splits.size();
    log.debug("Nb splits : {}", nbSplits);

    InputSplit[] sorted = new InputSplit[nbSplits];
    splits.toArray(sorted);
    Builder.sortSplits(sorted);

    int numTrees = Builder.getNbTrees(conf); // total number of trees

    TaskAttemptContext task = new TaskAttemptContext(conf, new TaskAttemptID());

    firstOutput = new MockContext(new Step1Mapper(), conf, task.getTaskAttemptID(), numTrees);

    firstIds = new int[nbSplits];
    sizes = new int[nbSplits];
    
    // to compute firstIds, process the splits in file order
    long slowest = 0; // duration of slowest map
    int firstId = 0;
    for (InputSplit split : splits) {
      int hp = ArrayUtils.indexOf(sorted, split); // hadoop's partition

      RecordReader<LongWritable, Text> reader = input.createRecordReader(split, task);
      reader.initialize(split, task);

      Step1Mapper mapper = new MockStep1Mapper(getTreeBuilder(), dataset, getSeed(),
                                               hp, nbSplits, numTrees);

      long time = System.currentTimeMillis();

      firstIds[hp] = firstId;

      while (reader.nextKeyValue()) {
        mapper.map(reader.getCurrentKey(), reader.getCurrentValue(), firstOutput);
        firstId++;
        sizes[hp]++;
      }

      mapper.cleanup(firstOutput);

      time = System.currentTimeMillis() - time;
      log.info("Duration : {}", DFUtils.elapsedTime(time));

      if (time > slowest) {
        slowest = time;
      }
    }

    log.info("Longest duration : {}", DFUtils.elapsedTime(slowest));
    return true;
  }

  @Override
  protected DecisionForest parseOutput(Job job, PredictionCallback callback) throws IOException, InterruptedException {
    Configuration conf = job.getConfiguration();
    
    DecisionForest forest = processOutput(firstOutput.getKeys(), firstOutput.getValues(), callback);

    if (isStep2(conf)) {
      Path forestPath = new Path(getOutputPath(conf), "step1.inter");
      FileSystem fs = forestPath.getFileSystem(conf);
      
      Node[] trees = new Node[forest.getTrees().size()];
      forest.getTrees().toArray(trees);
      InterResults.store(fs, forestPath, firstOutput.getKeys(), trees, sizes);

      log.info("***********");
      log.info("Second Step");
      log.info("***********");
      secondStep(conf, forestPath, callback);

      processOutput(secondOutput.getKeys(), secondOutput.getValues(), callback);
    }

    return forest;
  }

  /**
   * extract the decision forest and call the callback after correcting the instance ids
   */
  protected DecisionForest processOutput(TreeID[] keys, MapredOutput[] values, PredictionCallback callback) {
    List<Node> trees = Lists.newArrayList();

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
   */
  protected void secondStep(Configuration conf, Path forestPath, PredictionCallback callback)
      throws IOException, InterruptedException {
    JobContext jobContext = new JobContext(conf, new JobID());
    
    // retrieve the splits
    TextInputFormat input = new TextInputFormat();
    List<InputSplit> splits = input.getSplits(jobContext);
    
    int nbSplits = splits.size();
    log.debug("Nb splits : {}", nbSplits);

    InputSplit[] sorted = new InputSplit[nbSplits];
    splits.toArray(sorted);
    Builder.sortSplits(sorted);

    int numTrees = Builder.getNbTrees(conf); // total number of trees

    // compute the expected number of outputs
    int total = 0;
    for (int p = 0; p < nbSplits; p++) {
      total += Step2Mapper.nbConcerned(nbSplits, numTrees, p);
    }

    TaskAttemptContext task = new TaskAttemptContext(conf, new TaskAttemptID());

    secondOutput = new MockContext(new Step2Mapper(), conf, task.getTaskAttemptID(), numTrees);
    long slowest = 0; // duration of slowest map

    for (int partition = 0; partition < nbSplits; partition++) {
      
      InputSplit split = sorted[partition];
      RecordReader<LongWritable, Text> reader = input.createRecordReader(split, task);

      // load the output of the 1st step
      int nbConcerned = Step2Mapper.nbConcerned(nbSplits, numTrees, partition);
      TreeID[] fsKeys = new TreeID[nbConcerned];
      Node[] fsTrees = new Node[nbConcerned];

      FileSystem fs = forestPath.getFileSystem(conf);
      int numInstances = InterResults.load(fs, forestPath, nbSplits,
          numTrees, partition, fsKeys, fsTrees);

      Step2Mapper mapper = new Step2Mapper();
      mapper.configure(partition, dataset, fsKeys, fsTrees, numInstances);

      long time = System.currentTimeMillis();

      while (reader.nextKeyValue()) {
        mapper.map(reader.getCurrentKey(), reader.getCurrentValue(), secondOutput);
      }

      mapper.cleanup(secondOutput);

      time = System.currentTimeMillis() - time;
      log.info("Duration : {}", DFUtils.elapsedTime(time));

      if (time > slowest) {
        slowest = time;
      }
    }

    log.info("Longest duration : {}", DFUtils.elapsedTime(slowest));
  }

  /**
   * Special Step1Mapper that can be configured without using a Configuration
   * 
   */
  private static class MockStep1Mapper extends Step1Mapper {
    protected MockStep1Mapper(TreeBuilder treeBuilder, Dataset dataset, Long seed,
        int partition, int numMapTasks, int numTrees) {
      configure(false, true, treeBuilder, dataset);
      configure(seed, partition, numMapTasks, numTrees);
    }

  }

}
