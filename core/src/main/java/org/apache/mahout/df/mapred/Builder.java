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

package org.apache.mahout.df.mapred;

import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.common.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for Mapred DecisionForest builders. Takes care of storing the
 * parameters common to the mapred implementations.<br>
 * The child classes must implement at least :
 * <ul>
 * <li> void configureJob(JobConf) : to further configure the job before its
 * launch; and </li>
 * <li> DecisionForest parseOutput(JobConf, PredictionCallback) : in order to
 * convert the job outputs into a DecisionForest and its corresponding oob
 * predictions </li>
 * </ul>
 * 
 */
public abstract class Builder {

  private static final Logger log = LoggerFactory.getLogger(Builder.class);

  /** Tree Builder Component */
  protected final TreeBuilder treeBuilder;

  protected final Path dataPath;

  protected final Path datasetPath;

  protected final Long seed;

  protected final Configuration conf;

  private String outputDirName = "output";

  /**
   * Used only for DEBUG purposes. if false, the mappers doesn't output anything,
   * so the builder has nothing to process
   * 
   * @param conf
   * @return
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
   * @return null if no seed is available
   */
  public static Long getRandomSeed(Configuration conf) {
    String seed = conf.get("mahout.rf.random.seed");
    if (seed == null)
      return null;

    return Long.valueOf(seed);
  }

  /**
   * Sets the random seed value
   * 
   * @param conf
   * @param seed
   */
  private static void setRandomSeed(Configuration conf, long seed) {
    conf.setLong("mahout.rf.random.seed", seed);
  }

  public static TreeBuilder getTreeBuilder(Configuration conf) {
    String string = conf.get("mahout.rf.treebuilder");
    if (string == null)
      return null;

    return (TreeBuilder) StringUtils.fromString(string);
  }

  private static void setTreeBuilder(Configuration conf, TreeBuilder treeBuilder) {
    conf.set("mahout.rf.treebuilder", StringUtils.toString(treeBuilder));
  }

  /**
   * Get the number of trees for the map-reduce job. The default value is 100
   * 
   * @param conf
   * @return
   */
  public static int getNbTrees(Configuration conf) {
    return conf.getInt("mahout.rf.nbtrees", -1);
  }

  /**
   * Set the number of trees to grow for the map-reduce job
   * 
   * @param conf
   * @param nbTrees
   * @throws IllegalArgumentException if (nbTrees <= 0)
   */
  public static void setNbTrees(Configuration conf, int nbTrees) {
    if (nbTrees <= 0)
      throw new IllegalArgumentException("nbTrees should be greater than 0");

    conf.setInt("mahout.rf.nbtrees", nbTrees);
  }

  /**
   * Sets the Output directory name, will be creating in the working directory
   * @param name
   */
  public void setOutputDirName(String name) {
    outputDirName = name;
  }

  /**
   * Output Directory name
   * @param conf
   * @return
   * @throws IOException
   */
  public Path getOutputPath(Configuration conf) throws IOException {
    // the output directory is accessed only by this class, so use the default
    // file system
    FileSystem fs = FileSystem.get(conf);
    return new Path(fs.getWorkingDirectory(), outputDirName);
  }

  protected Builder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath,
      Long seed, Configuration conf) {
    this.treeBuilder = treeBuilder;
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
    this.seed = seed;
    this.conf = conf;
  }

  /**
   * Helper method. Get a path from the DistributedCache
   * 
   * @param job
   * @param index index of the path in the DistributedCache files
   * @return
   * @throws IOException
   */
  public static Path getDistributedCacheFile(JobConf job, int index)
      throws IOException {
    URI[] files = DistributedCache.getCacheFiles(job);

    if (files == null || files.length < index) {
      throw new IOException("path not found in the DistributedCache");
    }

    return new Path(files[index].getPath());
  }

  /**
   * Helper method. Load a Dataset stored in the DistributedCache
   * 
   * @param job
   * @return
   * @throws IOException
   */
  public static Dataset loadDataset(JobConf job) throws IOException {
    Path datasetPath = getDistributedCacheFile(job, 0);

    return Dataset.load(job, datasetPath);
  }

  /**
   * Used by the inheriting classes to configure the job
   * 
   * @param conf
   * @param nbTrees number of trees to grow
   * @param oobEstimate true, if oob error should be estimated
   * @throws IOException
   */
  protected abstract void configureJob(JobConf conf, int nbTrees,
      boolean oobEstimate) throws IOException;

  /**
   * Sequential implementation should override this method to simulate the job
   * execution
   */
  protected void runJob(JobConf job) throws IOException {
    JobClient.runJob(job);
  }

  /**
   * Parse the output files to extract the trees and pass the predictions to the
   * callback
   * 
   * @param job
   * @param callback can be null
   * @return
   * @throws IOException
   */
  protected abstract DecisionForest parseOutput(JobConf job,
      PredictionCallback callback) throws IOException;

  public DecisionForest build(int nbTrees, PredictionCallback callback) throws IOException {
    JobConf job = new JobConf(conf, Builder.class);

    Path outputPath = getOutputPath(job);
    FileSystem fs = outputPath.getFileSystem(job);

    // check the output
    if (fs.exists(outputPath))
      throw new IOException("Output path already exists : " + outputPath);

    if (seed != null)
      setRandomSeed(job, seed);
    setNbTrees(job, nbTrees);
    setTreeBuilder(job, treeBuilder);
    setOobEstimate(job, callback != null);

    // put the dataset into the DistributedCache
    DistributedCache.addCacheFile(datasetPath.toUri(), job);

    log.debug("Configuring the job...");
    configureJob(job, nbTrees, callback != null);

    log.debug("Running the job...");
    runJob(job);

    if (isOutput(job)) {
      log.debug("Parsing the output...");
      DecisionForest forest = parseOutput(job, callback);

      // delete the output path
      fs.delete(outputPath, true);

      return forest;
    }

    return null;
  }

  /**
   * sort the splits into order based on size, so that the biggest go first.<br>
   * This is the same code used by Hadoop's JobClient.
   * 
   * @param splits
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
        }
      }
    });
  }

}
