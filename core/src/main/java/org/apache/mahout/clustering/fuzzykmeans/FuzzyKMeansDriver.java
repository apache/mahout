/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.fuzzykmeans;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;

public class FuzzyKMeansDriver {

  private static final Log log = LogFactory.getLog(FuzzyKMeansDriver.class);

  private FuzzyKMeansDriver() {
  }

  public static void main(String[] args) {
    String input = args[0];
    String clusters = args[1];
    String output = args[2];
    String measureClass = args[3];
    double convergenceDelta = Double.parseDouble(args[4]);
    int maxIterations = Integer.parseInt(args[5]);
    int m = Integer.parseInt(args[6]);
    runJob(input, clusters, output, measureClass, convergenceDelta,
        maxIterations, 10,m);
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input the directory pathname for input points
   * @param clustersIn the directory pathname for initial & computed clusters
   * @param output the directory pathname for output points
   * @param measureClass the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param maxIterations the maximum number of iterations
   * @param numMapTasks the number of mapper tasks
   */
  public static void runJob(String input, String clustersIn, String output,
      String measureClass, double convergenceDelta, int maxIterations,
      int numMapTasks, int m) {

    boolean converged = false;
    int iteration = 0;
    String delta = Double.toString(convergenceDelta);

    // iterate until the clusters converge
    while (!converged && iteration < maxIterations) {
      log.info("Iteration {" + iteration + "}");

      // point the output to a new directory per iteration
      String clustersOut = output + File.separator + "clusters-" + iteration;
      converged = runIteration(input, clustersIn, clustersOut, measureClass,
          delta, numMapTasks, iteration, m);

      // now point the input to the old output directory
      clustersIn = output + File.separator + "clusters-" + iteration;
      iteration++;
    }

    // now actually cluster the points
    log.info("Clustering ");

    runClustering(input, clustersIn, output + File.separator + "points",
        measureClass, delta, numMapTasks, m);
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input the directory pathname for input points
   * @param clustersIn the directory pathname for iniput clusters
   * @param clustersOut the directory pathname for output clusters
   * @param measureClass the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param numMapTasks the number of map tasks
   * @param iterationNumber the iteration number that is going to run
   * @param m
   * @return true if the iteration successfully runs
   */
  private static boolean runIteration(String input, String clustersIn,
                                      String clustersOut, String measureClass, String convergenceDelta,
                                      int numMapTasks, int iterationNumber, int m) {

    JobConf conf = new JobConf(FuzzyKMeansJob.class);
    conf.setJobName("Fuzzy K Means{" + iterationNumber + "}");

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(clustersOut);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(FuzzyKMeansMapper.class);
    conf.setCombinerClass(FuzzyKMeansCombiner.class);
    conf.setReducerClass(FuzzyKMeansReducer.class);
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(numMapTasks);

    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(SoftCluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(SoftCluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(SoftCluster.CLUSTER_CONVERGENCE_KEY, convergenceDelta);
    conf.set(SoftCluster.M_KEY, String.valueOf(m));

    try {
      JobClient.runJob(conf);
      FileSystem fs = FileSystem.get(conf);
      return isConverged(clustersOut, conf, fs);
    } catch (IOException e) {
      log.warn(e.toString(), e);
      return true;
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input the directory pathname for input points
   * @param clustersIn the directory pathname for input clusters
   * @param output the directory pathname for output points
   * @param measureClass the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param numMapTasks the number of map tasks
   */
  private static void runClustering(String input, String clustersIn,
      String output, String measureClass, String convergenceDelta,
      int numMapTasks, float m) {

    JobConf conf = new JobConf(FuzzyKMeansDriver.class);
    conf.setJobName("Fuzzy K Means Clustering");

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(FuzzyKMeansClusterMapper.class);

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(0);
    conf.set(SoftCluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(SoftCluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(SoftCluster.CLUSTER_CONVERGENCE_KEY, convergenceDelta);
    conf.set(SoftCluster.M_KEY, String.valueOf(m));
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }

  /**
   * Return if all of the Clusters in the filePath have converged or not
   * 
   * @param filePath the file path to the single file containing the clusters
   * @param conf the JobConf
   * @param fs the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException if there was an IO error
   */
  private static boolean isConverged(String filePath, JobConf conf,
      FileSystem fs) throws IOException {

    Path clusterPath = new Path(filePath);
    List<Path> result = new ArrayList<Path>();

    PathFilter clusterFileFilter = new PathFilter() {
      public boolean accept(Path path) {
        return path.getName().startsWith("part");
      }
    };

    FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(
        clusterPath, clusterFileFilter)), clusterFileFilter);

    for (FileStatus match : matches) {
      result.add(fs.makeQualified(match.getPath()));
    }
    boolean converged = true;

    for (Path p : result) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, conf);
      Text key = new Text();
      Text value = new Text();

      while (converged && reader.next(key, value)) {
        converged = value.toString().startsWith("V");
      }
    }

    return converged;
  }
}
