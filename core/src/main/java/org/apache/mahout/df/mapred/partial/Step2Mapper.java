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
import java.net.URI;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.df.callback.SingleTreePredictions;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.mapreduce.partial.InterResults;
import org.apache.mahout.df.mapreduce.partial.TreeID;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Second step of PartialBuilder. Using the trees of the first step, computes
 * the oob predictions for each tree, except those of its own partition, on all
 * instancesof the partition.
 */
public class Step2Mapper extends MapReduceBase implements
    Mapper<LongWritable, Text, TreeID, MapredOutput> {

  private static final Logger log = LoggerFactory.getLogger(Step2Mapper.class);

  private TreeID[] keys;

  private Node[] trees;

  private SingleTreePredictions[] callbacks;

  private DataConverter converter;

  private int partition = -1;

  /** used by close() */
  private OutputCollector<TreeID, MapredOutput> output;

  /** num treated instances */
  private int instanceId;

  @Override
  public void configure(JobConf job) {
    // get the cached files' paths
    URI[] files;
    try {
      files = DistributedCache.getCacheFiles(job);
    } catch (IOException e) {
      throw new IllegalStateException("Exception while getting the cache files : ", e);
    }

    if (files == null || files.length < 2) {
      throw new IllegalArgumentException("missing paths from the DistributedCache");
    }

    Dataset dataset;
    try {
      Path datasetPath = new Path(files[0].getPath());
      dataset = Dataset.load(job, datasetPath);
    } catch (IOException e) {
      throw new IllegalStateException("Exception while loading the dataset : ", e);
    }

    int numMaps = job.getNumMapTasks();
    int p = job.getInt("mapred.task.partition", -1);

    // total number of trees in the forest
    int numTrees = Builder.getNbTrees(job);
    if (numTrees == -1) {
      throw new IllegalArgumentException("numTrees not found !");
    }

    int nbConcerned = nbConcerned(numMaps, numTrees, p);
    keys = new TreeID[nbConcerned];
    trees = new Node[nbConcerned];

    int numInstances;

    try {
      Path forestPath = new Path(files[1].getPath());
      FileSystem fs = forestPath.getFileSystem(job);
      numInstances = InterResults.load(fs, forestPath, numMaps, numTrees,
          p, keys, trees);

      log.debug("partition: " + p + "numInstances: " + numInstances);
    } catch (IOException e) {
      throw new IllegalStateException("Exception while loading the forest : ", e);
    }

    configure(p, dataset, keys, trees, numInstances);
  }

  /**
   * Compute the number of trees that need to classify the instances of this
   * mapper's partition
   * 
   * @param numMaps total number of map tasks
   * @param numTrees total number of trees in the forest
   * @param partition mapper's partition
   * @return
   */
  public static int nbConcerned(int numMaps, int numTrees, int partition) {
    if (partition < 0) {
      throw new IllegalArgumentException("partition < 0");
    }
    // the trees of the mapper's partition are not concerned
    return numTrees - Step1Mapper.nbTrees(numMaps, numTrees, partition);
  }

  /**
   * Useful for testing. Configures the mapper without using a JobConf<br>
   * TODO we don't need the keys partitions, the tree ids should suffice
   * 
   * @param partition mapper's partition
   * @param dataset
   * @param keys keys returned by the first step
   * @param trees trees returned by the first step
   * @param numInstances number of instances in the mapper's partition
   */
  public void configure(int partition, Dataset dataset, TreeID[] keys,
      Node[] trees, int numInstances) {
    this.partition = partition;
    if (partition < 0) {
      throw new IllegalArgumentException("Wrong partition id : " + partition);
    }

    converter = new DataConverter(dataset);

    if (keys.length != trees.length) {
      throw new IllegalArgumentException("keys.length != trees.length");
    }
    int nbConcerned = keys.length;

    this.keys = keys;
    this.trees = trees;

    // make sure the trees are not from this partition
    for (TreeID key : keys) {
      if (key.partition() == partition) {
        throw new IllegalArgumentException("a tree from this partition was found !");
      }
    }

    // init the callbacks
    callbacks = new SingleTreePredictions[nbConcerned];
    for (int index = 0; index < nbConcerned; index++) {
      callbacks[index] = new SingleTreePredictions(numInstances);
    }

  }

  @Override
  public void map(LongWritable key, Text value,
      OutputCollector<TreeID, MapredOutput> output, Reporter reporter)
      throws IOException {
    if (this.output == null) {
      this.output = output;
    }

    Instance instance = converter.convert(instanceId, value.toString());

    for (int index = 0; index < keys.length; index++) {
      int prediction = trees[index].classify(instance);
      callbacks[index].prediction(index, instanceId, prediction);
    }

    instanceId++;
  }

  @Override
  public void close() throws IOException {
    for (int index = 0; index < keys.length; index++) {
      TreeID key = new TreeID(partition, keys[index].treeId());
      output.collect(key, new MapredOutput(callbacks[index].getPredictions()));
    }
  }

}
