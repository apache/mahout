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

package org.apache.mahout.classifier.df.mapreduce.partial;

import java.io.IOException;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Simulates the Partial mapreduce implementation in a sequential manner. Must
 * receive a seed
 */
public class PartialSequentialBuilder extends PartialBuilder {

  private static final Logger log = LoggerFactory.getLogger(PartialSequentialBuilder.class);

  private MockContext firstOutput;

  private final Dataset dataset;

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
  protected void configureJob(Job job)
      throws IOException {
    Configuration conf = job.getConfiguration();
    
    int num = conf.getInt("mapred.map.tasks", -1);

    super.configureJob(job);

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

    /* first instance id in hadoop's order */
    //int[] firstIds = new int[nbSplits];
    /* partitions' sizes in hadoop order */
    int[] sizes = new int[nbSplits];
    
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

      //firstIds[hp] = firstId;

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
  protected DecisionForest parseOutput(Job job) throws IOException {
    return processOutput(firstOutput.getKeys(), firstOutput.getValues());
  }

  /**
   * extract the decision forest
   */
  protected static DecisionForest processOutput(TreeID[] keys, MapredOutput[] values) {
    List<Node> trees = Lists.newArrayList();

    for (int index = 0; index < keys.length; index++) {
      MapredOutput value = values[index];
      trees.add(value.getTree());
    }
    
    return new DecisionForest(trees);
  }

  /**
   * Special Step1Mapper that can be configured without using a Configuration
   * 
   */
  private static class MockStep1Mapper extends Step1Mapper {
    protected MockStep1Mapper(TreeBuilder treeBuilder, Dataset dataset, Long seed,
        int partition, int numMapTasks, int numTrees) {
      configure(false, treeBuilder, dataset);
      configure(seed, partition, numMapTasks, numTrees);
    }

  }

}
