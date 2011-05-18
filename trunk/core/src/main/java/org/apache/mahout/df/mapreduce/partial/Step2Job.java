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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.MapredOutput;
import org.apache.mahout.df.mapreduce.partial.Step0Job.Step0Output;
import org.apache.mahout.df.node.Node;

/**
 * 2nd step of the partial mapreduce builder. Computes the oob predictions using all the trees of the forest
 */
public class Step2Job {
  
  /** directory that will hold this job's output */
  private final Path outputPath;
  
  /** file that will contains the forest, passed to the maps */
  private final Path forestPath;
  
  /** file that contains the serialized dataset */
  private final Path datasetPath;
  
  /** directory that contains the data used in the first step */
  private final Path dataPath;
  
  /** partitions info in Hadoop's order */
  private final Step0Output[] partitions;
  
  /**
   * @param base
   *          base directory
   * @param dataPath
   *          data used in the first step
   * @param datasetPath
   * @param partitions
   *          partitions' infos in hadoop's order
   */
  public Step2Job(Path base, Path dataPath, Path datasetPath, Step0Output[] partitions) {
    this.outputPath = new Path(base, "step2.output");
    this.forestPath = new Path(base, "forest");
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
    this.partitions = partitions;
  }
  
  /**
   * Run the second step.
   * 
   * @param conf
   *          configuration
   * @param keys
   *          keys returned by the first step
   * @param trees
   *          trees returned by the first step
   * @param callback
   */
  public void run(Configuration conf, TreeID[] keys, Node[] trees, PredictionCallback callback) throws IOException,
                                                                                               ClassNotFoundException,
                                                                                               InterruptedException {
    if (callback == null) {
      // no need to launch the job
      return;
    }
    
    int numTrees = keys.length;
    
    // check the output
    if (outputPath.getFileSystem(conf).exists(outputPath)) {
      throw new IOException("Output path already exists : " + outputPath);
    }
    
    int[] sizes = Step0Output.extractSizes(partitions);
    
    InterResults.store(forestPath.getFileSystem(conf), forestPath, keys, trees, sizes);
    
    // needed by the mapper
    Builder.setNbTrees(conf, numTrees);
    
    // put the dataset and the forest into the DistributedCache
    // use setCacheFiles() to overwrite the first-step cache files
    URI[] files = {datasetPath.toUri(), forestPath.toUri()};
    DistributedCache.setCacheFiles(files, conf);
    
    Job job = new Job(conf);
    job.setJarByClass(Step2Job.class);
    
    FileInputFormat.setInputPaths(job, dataPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    
    job.setOutputKeyClass(TreeID.class);
    job.setOutputValueClass(MapredOutput.class);
    
    job.setMapperClass(Step2Mapper.class);
    job.setNumReduceTasks(0); // no reducers
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    // run the job
    if (job.waitForCompletion(true)) {
      parseOutput(job, callback);
    }
  }
  
  /**
   * Extracts the output and processes it
   * 
   * @param job
   * @param callback
   * @throws IOException
   */
  protected void parseOutput(Job job, PredictionCallback callback) throws IOException {
    Configuration conf = job.getConfiguration();
    
    int numMaps = Builder.getNumMaps(conf);
    int numTrees = Builder.getNbTrees(conf);
    
    // compute the total number of output values
    int total = 0;
    for (int partition = 0; partition < numMaps; partition++) {
      total += Step2Mapper.nbConcerned(numMaps, numTrees, partition);
    }
    
    int[] firstIds = Step0Output.extractFirstIds(partitions);
    PartialBuilder.processOutput(job, outputPath, firstIds, null, null, callback);
  }
}
