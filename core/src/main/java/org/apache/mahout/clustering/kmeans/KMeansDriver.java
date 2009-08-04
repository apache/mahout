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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.CommandLineUtil;
import org.apache.mahout.utils.HadoopUtil;
import org.apache.mahout.utils.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class KMeansDriver {

  /** The name of the directory used to output final results. */
  public static final String DEFAULT_OUTPUT_DIRECTORY = "/points";

  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  private KMeansDriver() {
  }

  /** @param args Expects 7 args and they all correspond to the order of the params in {@link #runJob} */
  public static void main(String[] args) throws ClassNotFoundException, IOException, IllegalAccessException,
      InstantiationException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option clustersOpt = obuilder
        .withLongName("clusters")
        .withRequired(true)
        .withArgument(abuilder.withName("clusters").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
                + "If k is also specified, then a random set of vectors will be selected and written out to this path first")
        .withShortName("c").create();

    Option kOpt = obuilder
        .withLongName("k")
        .withRequired(false)
        .withArgument(abuilder.withName("k").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen as the Centroid and written to the clusters output path.")
        .withShortName("k").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Path to put the output in").withShortName("o").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
        "If set, overwrite the output directory").withShortName("w").create();

    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Distance Measure to use.  Default is SquaredEuclidean").withShortName("m").create();

    Option convergenceDeltaOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(
        abuilder.withName("convergence").withMinimum(1).withMaximum(1).create()).withDescription(
        "The threshold below which the clusters are considered to be converged.  Default is 0.5").withShortName("d")
        .create();

    Option maxIterationsOpt = obuilder.withLongName("max").withRequired(false).withArgument(
        abuilder.withName("max").withMinimum(1).withMaximum(1).create()).withDescription(
        "The maximum number of iterations to perform.  Default is 20").withShortName("x").create();

    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Vector implementation class name.  Default is SparseVector.class").withShortName("v").create();

    Option numReduceTasksOpt = obuilder.withLongName("numReduce").withRequired(false).withArgument(
        abuilder.withName("numReduce").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of reduce tasks").withShortName("r").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(clustersOpt).withOption(outputOpt)
        .withOption(measureClassOpt).withOption(convergenceDeltaOpt).withOption(maxIterationsOpt).withOption(
            numReduceTasksOpt).withOption(kOpt).withOption(vectorClassOpt).withOption(overwriteOutput).withOption(
            helpOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      String input = cmdLine.getValue(inputOpt).toString();
      String clusters = cmdLine.getValue(clustersOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      String measureClass = SquaredEuclideanDistanceMeasure.class.getName();
      if (cmdLine.hasOption(measureClassOpt)) {
        measureClass = cmdLine.getValue(measureClassOpt).toString();
      }
      double convergenceDelta = 0.5;
      if (cmdLine.hasOption(convergenceDeltaOpt)) {
        convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      }

      Class<? extends Vector> vectorClass = cmdLine.hasOption(vectorClassOpt) == false ? SparseVector.class
          : (Class<? extends Vector>) Class.forName(cmdLine.getValue(vectorClassOpt).toString());

      int maxIterations = 20;
      if (cmdLine.hasOption(maxIterationsOpt)) {
        maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
      }
      int numReduceTasks = 2;
      if (cmdLine.hasOption(numReduceTasksOpt)) {
        numReduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      }
      if (cmdLine.hasOption(overwriteOutput) == true) {
        HadoopUtil.overwriteOutput(output);
      }
      if (cmdLine.hasOption(kOpt)) {
        clusters = RandomSeedGenerator
            .buildRandom(input, clusters, Integer.parseInt(cmdLine.getValue(kOpt).toString())).toString();
      }
      runJob(input, clusters, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, vectorClass);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
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
   */
  public static void runJob(String input, String clustersIn, String output, String measureClass,
      double convergenceDelta, int maxIterations, int numReduceTasks, Class<? extends Vector> vectorClass) {
    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: " + input + " Clusters In: " + clustersIn + " Out: " + output + " Distance: " + measureClass);
      log.info("convergence: " + convergenceDelta + " max Iterations: " + maxIterations + " num Reduce Tasks: "
          + numReduceTasks + " Input Vectors: " + vectorClass.getName());
    }
    boolean converged = false;
    int iteration = 0;
    while (!converged && iteration < maxIterations) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String clustersOut = output + "/clusters-" + iteration;
      converged = runIteration(input, clustersIn, clustersOut, measureClass, delta, numReduceTasks, iteration);
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
  private static boolean runIteration(String input, String clustersIn, String clustersOut, String measureClass,
      String convergenceDelta, int numReduceTasks, int iteration) {
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

    try {
      JobClient.runJob(conf);
      FileSystem fs = FileSystem.get(outPath.toUri(), conf);
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
   */
  private static void runClustering(String input, String clustersIn, String output, String measureClass,
      String convergenceDelta, Class<? extends Vector> vectorClass) {
    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: " + input + " Clusters In: " + clustersIn + " Out: " + output + " Distance: " + measureClass);
      log.info("convergence: " + convergenceDelta + " Input Vectors: " + vectorClass.getName());
    }
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(vectorClass);
    conf.setOutputKeyClass(Text.class);
    // the output is the cluster id
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(KMeansClusterMapper.class);
    conf.setNumReduceTasks(0);
    conf.set(Cluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(Cluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(Cluster.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }

  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath the file path to the single file containing the clusters
   * @param conf the JobConf
   * @param fs the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException if there was an IO error
   */
  private static boolean isConverged(String filePath, JobConf conf, FileSystem fs) throws IOException {
    FileStatus[] parts = fs.listStatus(new Path(filePath));
    for (FileStatus part : parts) {
      String name = part.getPath().getName();
      if (name.startsWith("part") && !name.endsWith(".crc")) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, part.getPath(), conf);
        Writable key;
        try {
          key = (Writable) reader.getKeyClass().newInstance();
        } catch (InstantiationException e) {// shouldn't happen
          log.error("Exception", e);
          throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
          log.error("Exception", e);
          throw new RuntimeException(e);
        }
        Cluster value = new Cluster();
        while (reader.next(key, value)) {
          if (value.isConverged() == false) {
            return false;
          }
        }
      }
    }
    return true;
  }
}
