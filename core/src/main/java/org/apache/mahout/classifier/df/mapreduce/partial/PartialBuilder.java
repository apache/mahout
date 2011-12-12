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

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
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
import java.util.Arrays;

/**
 * Builds a random forest using partial data. Each mapper uses only the data given by its InputSplit
 */
public class PartialBuilder extends Builder {

  public PartialBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath, Long seed) {
    this(treeBuilder, dataPath, datasetPath, seed, new Configuration());
  }
  
  public PartialBuilder(TreeBuilder treeBuilder,
                        Path dataPath,
                        Path datasetPath,
                        Long seed,
                        Configuration conf) {
    super(treeBuilder, dataPath, datasetPath, seed, conf);
  }

  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(PartialBuilder.class);
    
    FileInputFormat.setInputPaths(job, getDataPath());
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    job.setOutputKeyClass(TreeID.class);
    job.setOutputValueClass(MapredOutput.class);
    
    job.setMapperClass(Step1Mapper.class);
    job.setNumReduceTasks(0); // no reducers
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
  }
  
  @Override
  protected DecisionForest parseOutput(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    int numTrees = Builder.getNbTrees(conf);
    
    Path outputPath = getOutputPath(conf);
    
    TreeID[] keys = new TreeID[numTrees];
    Node[] trees = new Node[numTrees];
        
    processOutput(job, outputPath, keys, trees);
    
    return new DecisionForest(Arrays.asList(trees));
  }
  
  /**
   * Processes the output from the output path.<br>
   * 
   * @param outputPath
   *          directory that contains the output of the job
   * @param keys
   *          can be null
   * @param trees
   *          can be null
   * @throws java.io.IOException
   */
  protected static void processOutput(JobContext job,
                                      Path outputPath,
                                      TreeID[] keys,
                                      Node[] trees) throws IOException {
    Preconditions.checkArgument(keys == null && trees == null || keys != null && trees != null,
        "if keys is null, trees should also be null");
    Preconditions.checkArgument(keys == null || keys.length == trees.length, "keys.length != trees.length");

    Configuration conf = job.getConfiguration();

    FileSystem fs = outputPath.getFileSystem(conf);

    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);

    // read all the outputs
    int index = 0;
    for (Path path : outfiles) {
      for (Pair<TreeID,MapredOutput> record : new SequenceFileIterable<TreeID, MapredOutput>(path, conf)) {
        TreeID key = record.getFirst();
        MapredOutput value = record.getSecond();
        if (keys != null) {
          keys[index] = key;
        }
        if (trees != null) {
          trees[index] = value.getTree();
        }
        index++;
      }
    }

    // make sure we got all the keys/values
    if (keys != null && index != keys.length) {
      throw new IllegalStateException("Some key/values are missing from the output");
    }
  }
}
