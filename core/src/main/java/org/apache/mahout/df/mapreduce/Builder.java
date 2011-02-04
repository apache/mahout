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

package org.apache.mahout.df.mapreduce;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * Base class for Mapred DecisionForest builders. Takes care of storing the parameters common to the mapred
 * implementations.<br>
 * The child classes must implement at least :
 * <ul>
 * <li>void configureJob(Job) : to further configure the job before its launch; and</li>
 * <li>DecisionForest parseOutput(Job, PredictionCallback) : in order to convert the job outputs into a
 * DecisionForest and its corresponding oob predictions</li>
 * </ul>
 * 
 */
public abstract class Builder {
  
  private static final Logger log = LoggerFactory.getLogger(Builder.class);
  
  /** Tree Builder Component */
  private final TreeBuilder treeBuilder;
  
  private final Path dataPath;
  
  private final Path datasetPath;
  
  private final Long seed;
  
  private final Configuration conf;
  
  private String outputDirName = "output";
  
  protected Builder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath, Long seed, Configuration conf) {
    this.treeBuilder = treeBuilder;
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
    this.seed = seed;
    this.conf = new Configuration(conf);
  }
  
  protected TreeBuilder getTreeBuilder() {
    return treeBuilder;
  }
  
  protected Path getDataPath() {
    return dataPath;
  }
  
  protected Path getDatasetPath() {
    return datasetPath;
  }
  
  protected Long getSeed() {
    return seed;
  }
  
  /**
   * Return the value of "mapred.map.tasks". In case the 'local' runner is detected, returns 1
   * 
   * @param conf
   *          configuration
   * @return number of map tasks
   */
  public static int getNumMaps(Configuration conf) {
    // if we are in 'local' mode, correct the number of maps
    // or the mappers won't be able to compute the right indexes
    String tracker = conf.get("mapred.job.tracker", "local");
//    if ("local".equals(tracker)) {
//      log.warn("Hadoop running in 'local' mode, only one map task will be launched");
//      return 1;
//    }
    
    return conf.getInt("mapred.map.tasks", -1);
  }
  
  /**
   * Used only for DEBUG purposes. if false, the mappers doesn't output anything, so the builder has nothing
   * to process
   * 
   * @param conf
   *          configuration
   * @return true if the builder has to return output. false otherwise
   */
  protected static boolean isOutput(Configuration conf) {
    return conf.getBoolean("debug.mahout.rf.output", true);
  }
  
  protected static boolean isOobEstimate(Configuration conf) {
    return conf.getBoolean("mahout.rf.oob", false);
  }
  
  private static void setOobEstimate(Configuration conf, boolean value) {
    conf.setBoolean("mahout.rf.oob", value);
  }
  
  /**
   * Returns the random seed
   * 
   * @param conf
   *          configuration
   * @return null if no seed is available
   */
  public static Long getRandomSeed(Configuration conf) {
    String seed = conf.get("mahout.rf.random.seed");
    if (seed == null) {
      return null;
    }
    
    return Long.valueOf(seed);
  }
  
  /**
   * Sets the random seed value
   * 
   * @param conf
   *          configuration
   * @param seed
   *          random seed
   */
  private static void setRandomSeed(Configuration conf, long seed) {
    conf.setLong("mahout.rf.random.seed", seed);
  }
  
  public static TreeBuilder getTreeBuilder(Configuration conf) {
    String string = conf.get("mahout.rf.treebuilder");
    if (string == null) {
      return null;
    }
    
    return StringUtils.fromString(string);
  }
  
  private static void setTreeBuilder(Configuration conf, TreeBuilder treeBuilder) {
    conf.set("mahout.rf.treebuilder", StringUtils.toString(treeBuilder));
  }
  
  /**
   * Get the number of trees for the map-reduce job.
   * 
   * @param conf
   *          configuration
   * @return number of trees to build
   */
  public static int getNbTrees(Configuration conf) {
    return conf.getInt("mahout.rf.nbtrees", -1);
  }
  
  /**
   * Set the number of trees to grow for the map-reduce job
   * 
   * @param conf
   *          configuration
   * @param nbTrees
   *          number of trees to build
   * @throws IllegalArgumentException
   *           if (nbTrees <= 0)
   */
  public static void setNbTrees(Configuration conf, int nbTrees) {
    Preconditions.checkArgument(nbTrees > 0, "nbTrees should be greater than 0");

    conf.setInt("mahout.rf.nbtrees", nbTrees);
  }
  
  /**
   * Sets the Output directory name, will be creating in the working directory
   * 
   * @param name
   *          output dir. name
   */
  public void setOutputDirName(String name) {
    outputDirName = name;
  }
  
  /**
   * Output Directory name
   * 
   * @param conf
   *          configuration
   * @return output dir. path (%WORKING_DIRECTORY%/OUTPUT_DIR_NAME%)
   * @throws IOException
   *           if we cannot get the default FileSystem
   */
  public Path getOutputPath(Configuration conf) throws IOException {
    // the output directory is accessed only by this class, so use the default
    // file system
    FileSystem fs = FileSystem.get(conf);
    return new Path(fs.getWorkingDirectory(), outputDirName);
  }
  
  /**
   * Helper method. Get a path from the DistributedCache
   * 
   * @param conf
   *          configuration
   * @param index
   *          index of the path in the DistributedCache files
   * @return path from the DistributedCache
   * @throws IOException
   *           if no path is found
   */
  public static Path getDistributedCacheFile(Configuration conf, int index) throws IOException {
    URI[] files = DistributedCache.getCacheFiles(conf);
    
    if ((files == null) || (files.length <= index)) {
      throw new IOException("path not found in the DistributedCache");
    }
    
    return new Path(files[index].getPath());
  }
  
  /**
   * Helper method. Load a Dataset stored in the DistributedCache
   * 
   * @param conf
   *          configuration
   * @return loaded Dataset
   * @throws IOException
   *           if we cannot retrieve the Dataset path from the DistributedCache, or the Dataset could not be
   *           loaded
   */
  public static Dataset loadDataset(Configuration conf) throws IOException {
    Path datasetPath = getDistributedCacheFile(conf, 0);
    
    return Dataset.load(conf, datasetPath);
  }
  
  /**
   * Used by the inheriting classes to configure the job
   * 
   * @param job
   *          Hadoop's Job
   * @param nbTrees
   *          number of trees to grow
   * @param oobEstimate
   *          true, if oob error should be estimated
   * @throws IOException
   *           if anything goes wrong while configuring the job
   */
  protected abstract void configureJob(Job job, int nbTrees, boolean oobEstimate) throws IOException;
  
  /**
   * Sequential implementation should override this method to simulate the job execution
   * 
   * @param job
   *          Hadoop's job
   * @return true is the job succeeded
   */
  protected boolean runJob(Job job) throws ClassNotFoundException, IOException, InterruptedException {
    return job.waitForCompletion(true);
  }
  
  /**
   * Parse the output files to extract the trees and pass the predictions to the callback
   * 
   * @param job
   *          Hadoop's job
   * @param callback
   *          can be null
   * @return Built DecisionForest
   * @throws IOException
   *           if anything goes wrong while parsing the output
   */
  protected abstract DecisionForest parseOutput(Job job, PredictionCallback callback)
    throws IOException, ClassNotFoundException, InterruptedException;
  
  public DecisionForest build(int nbTrees, PredictionCallback callback)
      throws IOException, ClassNotFoundException, InterruptedException {
    // int numTrees = getNbTrees(conf);
    
    Path outputPath = getOutputPath(conf);
    FileSystem fs = outputPath.getFileSystem(conf);
    
    // check the output
    if (fs.exists(outputPath)) {
      throw new IOException("Output path already exists : " + outputPath);
    }
    
    if (seed != null) {
      setRandomSeed(conf, seed);
    }
    setNbTrees(conf, nbTrees);
    setTreeBuilder(conf, treeBuilder);
    setOobEstimate(conf, callback != null);
    
    // put the dataset into the DistributedCache
    DistributedCache.addCacheFile(datasetPath.toUri(), conf);
    
    Job job = new Job(conf, "decision forest builder");
    
    log.debug("Configuring the job...");
    configureJob(job, nbTrees, callback != null);
    
    log.debug("Running the job...");
    if (!runJob(job)) {
      log.error("Job failed!");
      return null;
    }
    
    if (isOutput(conf)) {
      log.debug("Parsing the output...");
      DecisionForest forest = parseOutput(job, callback);
      HadoopUtil.overwriteOutput(outputPath);
      return forest;
    }
    
    return null;
  }
  
  /**
   * sort the splits into order based on size, so that the biggest go first.<br>
   * This is the same code used by Hadoop's JobClient.
   * 
   * @param splits
   *          input splits
   */
  public static void sortSplits(InputSplit[] splits) {
    Arrays.sort(splits, new Comparator<InputSplit>() {
      @Override
      public int compare(InputSplit a, InputSplit b) {
        try {
          long left = a.getLength();
          long right = b.getLength();
          if (left == right) {
            return 0;
          } else if (left < right) {
            return 1;
          } else {
            return -1;
          }
        } catch (IOException ie) {
          throw new IllegalStateException("Problem getting input split size", ie);
        } catch (InterruptedException ie) {
          throw new IllegalStateException("Problem getting input split size", ie);
        }
      }
    });
  }
  
}
