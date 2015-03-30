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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * MapReduce implementation where each mapper loads a full copy of the data in-memory. The forest trees are
 * splitted across all the mappers
 */
public class InMemBuilder extends Builder {
  
  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath, Long seed, Configuration conf) {
    super(treeBuilder, dataPath, datasetPath, seed, conf);
  }
  
  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath) {
    this(treeBuilder, dataPath, datasetPath, null, new Configuration());
  }
  
  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(InMemBuilder.class);
    
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    // put the data in the DistributedCache
    DistributedCache.addCacheFile(getDataPath().toUri(), conf);
    
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(MapredOutput.class);
    
    job.setMapperClass(InMemMapper.class);
    job.setNumReduceTasks(0); // no reducers
    
    job.setInputFormatClass(InMemInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
  }
  
  @Override
  protected DecisionForest parseOutput(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    Map<Integer,MapredOutput> output = Maps.newHashMap();
    
    Path outputPath = getOutputPath(conf);
    FileSystem fs = outputPath.getFileSystem(conf);
    
    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);
    
    // import the InMemOutputs
    for (Path path : outfiles) {
      for (Pair<IntWritable,MapredOutput> record : new SequenceFileIterable<IntWritable,MapredOutput>(path, conf)) {
        output.put(record.getFirst().get(), record.getSecond());
      }
    }
    
    return processOutput(output);
  }
  
  /**
   * Process the output, extracting the trees
   */
  private static DecisionForest processOutput(Map<Integer,MapredOutput> output) {
    List<Node> trees = Lists.newArrayList();
    
    for (Map.Entry<Integer,MapredOutput> entry : output.entrySet()) {
      MapredOutput value = entry.getValue();
      trees.add(value.getTree());
    }
    
    return new DecisionForest(trees);
  }
}
