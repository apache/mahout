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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.df.DFUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.node.Node;

/**
 * MapReduce implementation where each mapper loads a full copy of the data
 * in-memory. The forest trees are splitted across all the mappers
 */
public class InMemBuilder extends Builder {

  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath,
      Long seed, Configuration conf) {
    super(treeBuilder, dataPath, datasetPath, seed, conf);
  }

  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath) {
    this(treeBuilder, dataPath, datasetPath, null, new Configuration());
  }

  @Override
  protected void configureJob(JobConf conf, int nbTrees, boolean oobEstimate)
      throws IOException {
    FileOutputFormat.setOutputPath(conf, getOutputPath(conf));

    // put the data in the DistributedCache
    DistributedCache.addCacheFile(dataPath.toUri(), conf);

    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(MapredOutput.class);

    conf.setMapperClass(InMemMapper.class);
    conf.setNumReduceTasks(0); // no reducers

    conf.setInputFormat(InMemInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
  }

  @Override
  protected DecisionForest parseOutput(JobConf conf, PredictionCallback callback)
      throws IOException {
    Map<Integer, MapredOutput> output = new HashMap<Integer, MapredOutput>();

    Path outputPath = getOutputPath(conf);
    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);

    // import the InMemOutputs
    IntWritable key = new IntWritable();
    MapredOutput value = new MapredOutput();

    for (Path path : outfiles) {
      Reader reader = new Reader(fs, path, conf);

      try {
        while (reader.next(key, value)) {
          output.put(key.get(), value.clone());
        }
      } finally {
        reader.close();
      }
    }

    return processOutput(output, callback);
  }

  /**
   * Process the output, extracting the trees and passing the predictions to the
   * callback
   * 
   * @param output
   * @param callback
   * @return
   */
  private static DecisionForest processOutput(Map<Integer, MapredOutput> output,
      PredictionCallback callback) {
    List<Node> trees = new ArrayList<Node>();

    for (Map.Entry<Integer, MapredOutput> entry : output.entrySet()) {
      MapredOutput value = entry.getValue();

      trees.add(value.getTree());

      if (callback != null) {
        int[] predictions = value.getPredictions();
        for (int index = 0; index < predictions.length; index++) {
          callback.prediction(entry.getKey(), index, predictions[index]);
        }
      }
    }

    return new DecisionForest(trees);
  }
}
