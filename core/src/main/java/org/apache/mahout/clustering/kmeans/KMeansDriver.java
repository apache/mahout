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
package org.apache.mahout.clustering.kmeans;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.mahout.matrix.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KMeansDriver {

  /**
   * The name of the directory used to output final results. 
   */
  public static final String DEFAULT_OUTPUT_DIRECTORY = "/points";
  
  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  private KMeansDriver() {
  }

  /**
   * 
   * @param args Expects 7 args and they all correspond to the order of the params in {@link #runJob}
   */
  public static void main(String[] args) throws ClassNotFoundException {
    String input = args[0];
    String clusters = args[1];
    String output = args[2];
    String measureClass = args[3];
    double convergenceDelta = Double.parseDouble(args[4]);
    int maxIterations = Integer.parseInt(args[5]);
    String vectorClassName = args[6];
    Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
    runJob(input, clusters, output, measureClass, convergenceDelta,
        maxIterations, 2, vectorClass);
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
   * @param numReduceTasks the number of reducers
   * @param vectorClass
   */
  public static void runJob(String input, String clustersIn, String output,
                            String measureClass, double convergenceDelta, int maxIterations,
                            int numReduceTasks, Class<? extends Vector> vectorClass) {
    // iterate until the clusters converge
    boolean converged = false;
    int iteration = 0;
    String delta = Double.toString(convergenceDelta);

    while (!converged && iteration < maxIterations) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String clustersOut = output + "/clusters-" + iteration;
      converged = runIteration(input, clustersIn, clustersOut, measureClass,
          delta, numReduceTasks, iteration);
      // now point the input to the old output directory
      clustersIn = output + "/clusters-" + iteration;
      iteration++;
    }
    // now actually cluster the points
    log.info("Clustering ");
    runClustering(input, clustersIn, output + DEFAULT_OUTPUT_DIRECTORY, measureClass, delta, vectorClass);
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input the directory pathname for input points
   * @param clustersIn the directory pathname for input clusters
   * @param clustersOut the directory pathname for output clusters
   * @param measureClass the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param numReduceTasks the number of reducer tasks
   * @param iteration The iteration number
   * @return true if the iteration successfully runs
   */
  private static boolean runIteration(String input, String clustersIn,
                                      String clustersOut, String measureClass, String convergenceDelta,
                                      int numReduceTasks, int iteration) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(KMeansInfo.class);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Cluster.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(clustersOut);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setMapperClass(KMeansMapper.class);
    conf.setCombinerClass(KMeansCombiner.class);
    conf.setReducerClass(KMeansReducer.class);
    conf.setNumReduceTasks(numReduceTasks);
    conf.set(Cluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(Cluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(Cluster.CLUSTER_CONVERGENCE_KEY, convergenceDelta);
    conf.setInt(Cluster.ITERATION_NUMBER, iteration);
    
    client.setConf(conf);
    try {
      JobClient.runJob(conf);
      FileSystem fs = FileSystem.get(outPath.toUri(), conf);
      return isConverged(clustersOut + "/part-00000", conf, fs);
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
   */
  private static void runClustering(String input, String clustersIn,
      String output, String measureClass, String convergenceDelta, Class<? extends Vector> vectorClass) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(vectorClass);
    conf.setOutputKeyClass(Text.class);
    //the output is the cluster id
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(KMeansClusterMapper.class);
    conf.setNumReduceTasks(0);
    conf.set(Cluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(Cluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(Cluster.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    client.setConf(conf);
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
   * @param conf     the JobConf
   * @param fs       the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException if there was an IO error
   */
 private static boolean isConverged(String filePath, JobConf conf, FileSystem fs)
          throws IOException {
    Path outPart = new Path(filePath);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, outPart, conf);
    Text key = new Text();
    Cluster value = new Cluster();
    boolean converged = true;
    while (converged && reader.next(key, value)) {
      converged = value.isConverged();
    }
    return converged;
  }
}
