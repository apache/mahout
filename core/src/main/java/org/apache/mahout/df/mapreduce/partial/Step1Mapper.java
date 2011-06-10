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
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.Bagging;
import org.apache.mahout.df.callback.SingleTreePredictions;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.MapredMapper;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * First step of the Partial Data Builder. Builds the trees using the data available in the InputSplit.
 * Predict the oob classes for each tree in its growing partition (input split).
 */
public class Step1Mapper extends MapredMapper<LongWritable,Text,TreeID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(Step1Mapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  private Random rng;
  
  /** number of trees to be built by this mapper */
  private int nbTrees;
  
  /** id of the first tree */
  private int firstTreeId;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  public int getFirstTreeId() {
    return firstTreeId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    configure(Builder.getRandomSeed(conf), conf.getInt("mapred.task.partition", -1),
      Builder.getNumMaps(conf), Builder.getNbTrees(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param seed
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
   */
  protected void configure(Long seed, int partition, int numMapTasks, int numTrees) {
    converter = new DataConverter(getDataset());
    
    // prepare random-numders generator
    log.debug("seed : {}", seed);
    if (seed == null) {
      rng = RandomUtils.getRandom();
    } else {
      rng = RandomUtils.getRandom(seed);
    }
    
    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    
    // compute number of trees to build
    nbTrees = nbTrees(numMapTasks, numTrees, partition);
    
    // compute first tree id
    firstTreeId = 0;
    for (int p = 0; p < partition; p++) {
      firstTreeId += nbTrees(numMapTasks, numTrees, p);
    }
    
    log.debug("partition : {}", partition);
    log.debug("nbTrees : {}", nbTrees);
    log.debug("firstTreeId : {}", firstTreeId);
  }
  
  /**
   * Compute the number of trees for a given partition. The first partition (0) may be longer than the rest of
   * partition because of the remainder.
   * 
   * @param numMaps
   *          total number of maps (partitions)
   * @param numTrees
   *          total number of trees to build
   * @param partition
   *          partition to compute the number of trees for
   */
  public static int nbTrees(int numMaps, int numTrees, int partition) {
    int nbTrees = numTrees / numMaps;
    if (partition == 0) {
      nbTrees += numTrees - nbTrees * numMaps;
    }
    
    return nbTrees;
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert((int) key.get(), value.toString()));
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // prepare the data
    log.debug("partition: {} numInstances: {}", partition, instances.size());
    
    Data data = new Data(getDataset(), instances);
    Bagging bagging = new Bagging(getTreeBuilder(), data);
    
    TreeID key = new TreeID();
    
    log.debug("Building {} trees", nbTrees);
    SingleTreePredictions callback = null;
    int[] predictions = null;
    for (int treeId = 0; treeId < nbTrees; treeId++) {
      log.debug("Building tree number : {}", treeId);
      if (isOobEstimate() && !isNoOutput()) {
        callback = new SingleTreePredictions(data.size());
        predictions = callback.getPredictions();
      }
      
      Node tree = bagging.build(treeId, rng, callback);
      
      key.set(partition, firstTreeId + treeId);
      
      if (!isNoOutput()) {
        MapredOutput emOut = new MapredOutput(tree, predictions);
        context.write(key, emOut);
      }
    }
  }
  
}
