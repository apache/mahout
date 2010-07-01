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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class KMeansDriver {

  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  private KMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option clustersOpt = DefaultOptionCreator.clustersInOption().withDescription(
        "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected"
            + " and written out to this path first")
        .create();
    Option kOpt = DefaultOptionCreator.kOption().withDescription(
        "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option overwriteOutput = DefaultOptionCreator.overwriteOption().create();
    Option measureClassOpt = DefaultOptionCreator.distanceMeasureOption().create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption().create();
    Option maxIterationsOpt = DefaultOptionCreator.maxIterationsOption().create();
    Option numReduceTasksOpt = DefaultOptionCreator.numReducersOption().create();
    Option clusteringOpt = DefaultOptionCreator.clusteringOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = new GroupBuilder().withName("Options").withOption(inputOpt).withOption(clustersOpt)
        .withOption(outputOpt).withOption(measureClassOpt).withOption(convergenceDeltaOpt)
        .withOption(maxIterationsOpt).withOption(numReduceTasksOpt).withOption(kOpt).withOption(overwriteOutput)
        .withOption(helpOpt).withOption(clusteringOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path clusters = new Path(cmdLine.getValue(clustersOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());
      String measureClass = cmdLine.getValue(measureClassOpt).toString();
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
      int numReduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      if (cmdLine.hasOption(kOpt)) {
        clusters = RandomSeedGenerator.buildRandom(input, clusters,
                                                   Integer.parseInt(cmdLine.getValue(kOpt).toString()));
      }
      runJob(input, clusters, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, cmdLine
          .hasOption(clusteringOpt));
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void runJob(Path input, Path clustersIn, Path output, String measureClass, double convergenceDelta,
      int maxIterations, int numReduceTasks, boolean runClustering) throws IOException, InterruptedException,
      ClassNotFoundException {
    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}",
               new Object[] { input, clustersIn, output, measureClass });
      log.info("convergence: {} max Iterations: {} num Reduce Tasks: {} Input Vectors: {}",
               new Object[] { convergenceDelta, maxIterations, numReduceTasks, VectorWritable.class.getName() });
    }
    boolean converged = false;
    int iteration = 1;
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input, clustersIn, clustersOut, measureClass, delta, numReduceTasks);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }
    if (runClustering) {
      // now actually cluster the points
      log.info("Clustering ");
      runClustering(input, clustersIn, new Path(output, Cluster.CLUSTERED_POINTS_DIR), measureClass, delta);
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param clustersOut
   *          the directory pathname for output clusters
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param numReduceTasks
   *          the number of reducer tasks
   * @return true if the iteration successfully runs
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private static boolean runIteration(Path input,
                                      Path clustersIn,
                                      Path clustersOut,
                                      String measureClass,
                                      String convergenceDelta,
                                      int numReduceTasks)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(KMeansInfo.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Cluster.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(KMeansMapper.class);
    job.setCombinerClass(KMeansCombiner.class);
    job.setReducerClass(KMeansReducer.class);
    job.setNumReduceTasks(numReduceTasks);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    HadoopUtil.overwriteOutput(clustersOut);
    job.waitForCompletion(true);
    FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);

    return isConverged(clustersOut, conf, fs);
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private static void runClustering(Path input,
                                    Path clustersIn,
                                    Path output,
                                    String measureClass,
                                    String convergenceDelta)
    throws IOException, InterruptedException, ClassNotFoundException {
    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}",
               new Object[] { input, clustersIn, output, measureClass });
      log.info("convergence: {} Input Vectors: {}", convergenceDelta, VectorWritable.class.getName());
    }
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    HadoopUtil.overwriteOutput(output);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(KMeansClusterMapper.class);
    job.setNumReduceTasks(0);

    job.waitForCompletion(true);
  }

  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @param conf
   *          the JobConf
   * @param fs
   *          the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   */
  private static boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {
    FileStatus[] parts = fs.listStatus(filePath);
    for (FileStatus part : parts) {
      String name = part.getPath().getName();
      if (name.startsWith("part") && !name.endsWith(".crc")) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, part.getPath(), conf);
        try {
          Writable key = (Writable) reader.getKeyClass().newInstance();
          Cluster value = new Cluster();
          while (reader.next(key, value)) {
            if (!value.isConverged()) {
              return false;
            }
          }
        } catch (InstantiationException e) { // shouldn't happen
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } catch (IllegalAccessException e) {
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } finally {
          reader.close();
        }
      }
    }
    return true;
  }
}
