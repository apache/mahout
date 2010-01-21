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

package org.apache.mahout.df.mapred.inmem;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.StringUtils;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.Bagging;
import org.apache.mahout.df.callback.SingleTreePredictions;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapred.MapredMapper;
import org.apache.mahout.df.mapred.inmem.InMemInputFormat.InMemInputSplit;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * In-memory mapper that grows the trees using a full copy of the data loaded
 * in-memory. The number of trees to grow is determined by the current
 * InMemInputSplit.
 */
public class InMemMapper extends MapredMapper implements
    Mapper<IntWritable, NullWritable, IntWritable, MapredOutput> {

  private static final Logger log = LoggerFactory.getLogger(InMemMapper.class);

  private Bagging bagging;

  private Random rng;

  private Data data;

  /**
   * Load the training data
   * 
   * @param conf
   * @return
   * @throws RuntimeException if the data could not be loaded
   */
  private static Data loadData(JobConf conf, Dataset dataset) throws IOException {
    Path dataPath = Builder.getDistributedCacheFile(conf, 1);
    FileSystem fs = FileSystem.get(dataPath.toUri(), conf);
    return DataLoader.loadData(dataset, fs, dataPath);
  }

  @Override
  public void configure(JobConf conf) {
    super.configure(conf);

    log.info("Loading the data...");
    try {
      data = loadData(conf, getDataset());
    } catch (IOException e) {
    throw new IllegalStateException("Exception caught while loading the data: "
        + StringUtils.stringifyException(e));
    }
    log.info("Data loaded : {} instances", data.size());

    bagging = new Bagging(getTreeBuilder(), data);
  }

  @Override
  public void map(IntWritable key, NullWritable value,
      OutputCollector<IntWritable, MapredOutput> output, Reporter reporter)
      throws IOException {
    map(key, output, (InMemInputSplit) reporter.getInputSplit());
  }

  public void map(IntWritable key,
      OutputCollector<IntWritable, MapredOutput> output, InMemInputSplit split)
      throws IOException {

    SingleTreePredictions callback = null;
    int[] predictions = null;

    if (isOobEstimate() && !isNoOutput()) {
      callback = new SingleTreePredictions(data.size());
      predictions = callback.getPredictions();
    }

    initRandom(split);

    log.debug("Building...");
    Node tree = bagging.build(key.get(), rng, callback);

    if (!isNoOutput()) {
      log.debug("Outputing...");
      MapredOutput mrOut = new MapredOutput(tree, predictions);

      output.collect(key, mrOut);
    }
  }

  protected void initRandom(InMemInputSplit split) {
    if (rng == null) { // first execution of this mapper
      Long seed = split.getSeed();
      log.debug("Initialising rng with seed {}: ", seed);

      if (seed == null)
        rng = RandomUtils.getRandom();
      else
        rng = RandomUtils.getRandom(seed);
    }
  }

}
