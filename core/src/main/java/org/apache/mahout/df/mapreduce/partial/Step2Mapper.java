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
import java.net.URI;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.df.callback.SingleTreePredictions;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * Second step of PartialBuilder. Using the trees of the first step, computes the oob predictions for each
 * tree, except those of its own partition, on all instancesof the partition.
 */
public class Step2Mapper extends Mapper<LongWritable,Text,TreeID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(Step2Mapper.class);
  
  private TreeID[] keys;
  private Node[] trees;
  private SingleTreePredictions[] callbacks;
  private DataConverter converter;
  private int partition = -1;
  /** num treated instances */
  private int instanceId;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    
    // get the cached files' paths
    URI[] files = DistributedCache.getCacheFiles(conf);
    
    log.info("DistributedCache.getCacheFiles(): {}", ArrayUtils.toString(files));
    
    Preconditions.checkArgument(files != null && files.length >= 2, "missing paths from the DistributedCache");
    
    Path datasetPath = new Path(files[0].getPath());
    Dataset dataset = Dataset.load(conf, datasetPath);
    
    int numMaps = Builder.getNumMaps(conf);
    int p = conf.getInt("mapred.task.partition", -1);
    
    // total number of trees in the forest
    int numTrees = Builder.getNbTrees(conf);
    Preconditions.checkArgument(numTrees != -1, "numTrees not found !");
    
    int nbConcerned = nbConcerned(numMaps, numTrees, p);
    keys = new TreeID[nbConcerned];
    trees = new Node[nbConcerned];
    
    Path forestPath = new Path(files[1].getPath());
    FileSystem fs = forestPath.getFileSystem(conf);
    int numInstances = InterResults.load(fs, forestPath, numMaps, numTrees, p, keys, trees);
    
    log.debug("partition: {} numInstances: {}", p, numInstances);
    configure(p, dataset, keys, trees, numInstances);
  }
  
  /**
   * Compute the number of trees that need to classify the instances of this mapper's partition
   * 
   * @param numMaps
   *          total number of map tasks
   * @param numTrees
   *          total number of trees in the forest
   * @param partition
   *          mapper's partition
   */
  public static int nbConcerned(int numMaps, int numTrees, int partition) {
    Preconditions.checkArgument(partition >= 0, "partition < 0");
    // the trees of the mapper's partition are not concerned
    return numTrees - Step1Mapper.nbTrees(numMaps, numTrees, partition);
  }
  
  /**
   * Useful for testing. Configures the mapper without using a Configuration<br>
   * TODO we don't need the keys partitions, the tree ids should suffice
   * 
   * @param partition
   *          mapper's partition
   * @param dataset
   * @param keys
   *          keys returned by the first step
   * @param trees
   *          trees returned by the first step
   * @param numInstances
   *          number of instances in the mapper's partition
   */
  public void configure(int partition, Dataset dataset, TreeID[] keys, Node[] trees, int numInstances) {
    this.partition = partition;
    Preconditions.checkArgument(partition >= 0, "Wrong partition id : " + partition);
    
    converter = new DataConverter(dataset);

    Preconditions.checkArgument(keys.length == trees.length, "keys.length != trees.length");
    int nbConcerned = keys.length;
    
    this.keys = keys;
    this.trees = trees;
    
    // make sure the trees are not from this partition
    for (TreeID key : keys) {
      Preconditions.checkArgument(key.partition() != partition, "a tree from this partition was found !");
    }
    
    // init the callbacks
    callbacks = new SingleTreePredictions[nbConcerned];
    for (int index = 0; index < nbConcerned; index++) {
      callbacks[index] = new SingleTreePredictions(numInstances);
    }
    
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    
    Instance instance = converter.convert(instanceId, value.toString());
    
    for (int index = 0; index < keys.length; index++) {
      int prediction = trees[index].classify(instance);
      callbacks[index].prediction(index, instanceId, prediction);
    }
    
    instanceId++;
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    for (int index = 0; index < keys.length; index++) {
      TreeID key = new TreeID(partition, keys[index].treeId());
      context.write(key, new MapredOutput(callbacks[index].getPredictions()));
    }
  }
  
}
