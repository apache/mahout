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

package org.apache.mahout.classifier.df.mapreduce.inmem;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.mahout.classifier.df.Bagging;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredMapper;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat.InMemInputSplit;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

/**
 * In-memory mapper that grows the trees using a full copy of the data loaded in-memory. The number of trees
 * to grow is determined by the current InMemInputSplit.
 */
public class InMemMapper extends MapredMapper<IntWritable,NullWritable,IntWritable,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(InMemMapper.class);
  
  private Bagging bagging;
  
  private Random rng;

  /**
   * Load the training data
   */
  private static Data loadData(Configuration conf, Dataset dataset) throws IOException {
    Path dataPath = Builder.getDistributedCacheFile(conf, 1);
    FileSystem fs = FileSystem.get(dataPath.toUri(), conf);
    return DataLoader.loadData(dataset, fs, dataPath);
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    log.info("Loading the data...");
    Data data = loadData(conf, getDataset());
    log.info("Data loaded : {} instances", data.size());
    
    bagging = new Bagging(getTreeBuilder(), data);
  }
  
  @Override
  protected void map(IntWritable key,
                     NullWritable value,
                     Context context) throws IOException, InterruptedException {
    map(key, context);
  }
  
  void map(IntWritable key, Context context) throws IOException, InterruptedException {
    
    initRandom((InMemInputSplit) context.getInputSplit());
    
    log.debug("Building...");
    Node tree = bagging.build(rng);
    
    if (isOutput()) {
      log.debug("Outputing...");
      MapredOutput mrOut = new MapredOutput(tree);
      
      context.write(key, mrOut);
    }
  }
  
  void initRandom(InMemInputSplit split) {
    if (rng == null) { // first execution of this mapper
      Long seed = split.getSeed();
      log.debug("Initialising rng with seed : {}", seed);
      rng = seed == null ? RandomUtils.getRandom() : RandomUtils.getRandom(seed);
    }
  }
  
}
