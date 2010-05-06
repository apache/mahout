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

package org.apache.mahout.clustering.fuzzykmeans;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FuzzyKMeansDriver {

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansDriver.class);
  
  private FuzzyKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option clustersOpt = obuilder.withLongName("clusters").withRequired(true).withArgument(
        abuilder.withName("clusters").withMinimum(1).withMaximum(1).create()).withDescription(
        "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected" + " and written out to this path first")
        .withShortName("c").create();

    Option kOpt = obuilder.withLongName("k").withRequired(false).withArgument(
        abuilder.withName("k").withMinimum(1).withMaximum(1).create()).withDescription(
        "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters output path.").withShortName("k").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription("The Path to put the output in")
        .withShortName("o").create();

    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Distance Measure to use.  Default is SquaredEuclidean").withShortName("dm").create();

    Option convergenceDeltaOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(
        abuilder.withName("convergence").withMinimum(1).withMaximum(1).create()).withDescription(
        "The threshold below which the clusters are considered to be converged.  Default is 0.5").withShortName("d").create();

    Option maxIterationsOpt = obuilder.withLongName("max").withRequired(false).withArgument(
        abuilder.withName("max").withMinimum(1).withMaximum(1).create()).withDescription(
        "The maximum number of iterations to perform.  Default is 20").withShortName("x").create();

    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Vector implementation class name.  Default is RandomAccessSparseVector.class").withShortName("v").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
        "If set, overwrite the output directory").withShortName("w").create();

    Option mOpt = obuilder.withLongName("m").withRequired(true).withArgument(
        abuilder.withName("m").withMinimum(1).withMaximum(1).create()).withDescription(
        "coefficient normalization factor, must be greater than 1").withShortName("m").create();

    Option numReduceTasksOpt = obuilder.withLongName("numReduce").withRequired(false).withArgument(
        abuilder.withName("numReduce").withMinimum(1).withMaximum(1).create()).withDescription("The number of reduce tasks")
        .withShortName("r").create();

    Option numMapTasksOpt = obuilder.withLongName("numMap").withRequired(false).withArgument(
        abuilder.withName("numMap").withMinimum(1).withMaximum(1).create()).withDescription("The number of map tasks")
        .withShortName("u").create();

    Option clusteringOpt = obuilder.withLongName("clustering").withRequired(false).withDescription(
        "If true, run clustering after the iterations have taken place").withShortName("cl").create();

    Option emitMostLikelyOpt = obuilder.withLongName("emitMostLikely").withRequired(false).withShortName("e").withArgument(
        abuilder.withName("emitMostLikely").withMinimum(1).withMaximum(1).create()).withDescription(
        "True if clustering emits most likely point only, false for threshold clustering").create();

    Option thresholdOpt = obuilder.withLongName("threshold").withRequired(false).withShortName("t").withArgument(
        abuilder.withName("threshold").withMinimum(1).withMaximum(1).create()).withDescription("The pdf threshold").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(clustersOpt).withOption(outputOpt).withOption(
        measureClassOpt).withOption(convergenceDeltaOpt).withOption(maxIterationsOpt).withOption(kOpt).withOption(mOpt).withOption(
        vectorClassOpt).withOption(overwriteOutput).withOption(helpOpt).withOption(emitMostLikelyOpt).withOption(thresholdOpt)
        .create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path clusters = new Path(cmdLine.getValue(clustersOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());
      String measureClass = SquaredEuclideanDistanceMeasure.class.getName();
      if (cmdLine.hasOption(measureClassOpt)) {
        measureClass = cmdLine.getValue(measureClassOpt).toString();
      }
      double convergenceDelta = 0.5;
      if (cmdLine.hasOption(convergenceDeltaOpt)) {
        convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      }
      float m = Float.parseFloat(cmdLine.getValue(mOpt).toString());

      // Class<? extends Vector> vectorClass = cmdLine.hasOption(vectorClassOpt) == false ?
      // RandomAccessSparseVector.class
      // : (Class<? extends Vector>) Class.forName(cmdLine.getValue(vectorClassOpt).toString());

      int numReduceTasks = 10;
      if (cmdLine.hasOption(numReduceTasksOpt)) {
        numReduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      }

      int numMapTasks = 50;
      if (cmdLine.hasOption(numMapTasksOpt)) {
        numMapTasks = Integer.parseInt(cmdLine.getValue(numMapTasksOpt).toString());
      }

      int maxIterations = 20;
      if (cmdLine.hasOption(maxIterationsOpt)) {
        maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
      }

      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }

      if (cmdLine.hasOption(kOpt)) {
        clusters = RandomSeedGenerator.buildRandom(input, clusters, Integer.parseInt(cmdLine.getValue(kOpt).toString()));
      }

      boolean emitMostLikely = true;
      if (cmdLine.hasOption(emitMostLikelyOpt)) {
        emitMostLikely = Boolean.parseBoolean(cmdLine.getValue(emitMostLikelyOpt).toString());
      }
      double threshold = 0;
      if (cmdLine.hasOption(thresholdOpt)) {
        threshold = Double.parseDouble(cmdLine.getValue(thresholdOpt).toString());
      }
      runJob(input, clusters, output, measureClass, convergenceDelta, maxIterations, numMapTasks, numReduceTasks, m, cmdLine
          .hasOption(clusteringOpt), emitMostLikely, threshold);

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
   * @param numMapTasks
   *          the number of mapper tasks
   * @param numReduceTasks
   *          the number of reduce tasks
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runClustering 
   *          true if points are to be clustered after iterations complete
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   */
  public static void runJob(Path input, Path clustersIn, Path output, String measureClass, double convergenceDelta,
      int maxIterations, int numMapTasks, int numReduceTasks, float m, boolean runClustering, boolean emitMostLikely,
      double threshold) {

    boolean converged = false;
    int iteration = 1;

    // iterate until the clusters converge
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);

      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input, clustersIn, clustersOut, measureClass, convergenceDelta, numMapTasks, numReduceTasks,
          iteration, m);

      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }

    // now actually cluster the points
    log.info("Clustering ");
    runClustering(input, clustersIn, new Path(output, Cluster.CLUSTERED_POINTS_DIR), measureClass, convergenceDelta, numMapTasks, m,
        emitMostLikely, threshold);
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for iniput clusters
   * @param clustersOut
   *          the directory pathname for output clusters
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param numMapTasks
   *          the number of map tasks
   * @param iterationNumber
   *          the iteration number that is going to run
   * @param m
   *          the fuzzification factor - see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @return true if the iteration successfully runs
   */
  private static boolean runIteration(Path input, Path clustersIn, Path clustersOut, String measureClass,
      double convergenceDelta, int numMapTasks, int numReduceTasks, int iterationNumber, float m) {

    JobConf conf = new JobConf(FuzzyKMeansDriver.class);
    conf.setJobName("Fuzzy K Means{" + iterationNumber + '}');

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(FuzzyKMeansInfo.class);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(SoftCluster.class);

    FileInputFormat.setInputPaths(conf, input);
    FileOutputFormat.setOutputPath(conf, clustersOut);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.setMapperClass(FuzzyKMeansMapper.class);
    conf.setCombinerClass(FuzzyKMeansCombiner.class);
    conf.setReducerClass(FuzzyKMeansReducer.class);
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(numReduceTasks);

    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    // these values don't matter during iterations as only used for clustering if requested
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(true));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(0));

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");

    try {
      JobClient.runJob(conf);
      FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);
      return isConverged(clustersOut, conf, fs);
    } catch (IOException e) {
      log.warn(e.toString(), e);
      return true;
    }
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
   * @param numMapTasks
   *          the number of map tasks
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   */
  private static void runClustering(Path input, Path clustersIn, Path output, String measureClass, double convergenceDelta,
      int numMapTasks, float m, boolean emitMostLikely, double threshold) {

    JobConf conf = new JobConf(FuzzyKMeansDriver.class);
    conf.setJobName("Fuzzy K Means Clustering");

    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(conf, input);
    FileOutputFormat.setOutputPath(conf, output);

    conf.setMapperClass(FuzzyKMeansClusterMapper.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(0);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(threshold));
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }

  /**
   * Return if all of the Clusters in the filePath have converged or not
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

    Path clusterPath = new Path(filePath, "*");
    List<Path> result = new ArrayList<Path>();

    PathFilter clusterFileFilter = new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return path.getName().startsWith("part");
      }
    };

    FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(clusterPath, clusterFileFilter)), clusterFileFilter);

    for (FileStatus match : matches) {
      result.add(fs.makeQualified(match.getPath()));
    }
    boolean converged = true;

    for (Path p : result) {

      SequenceFile.Reader reader = null;

      try {
        reader = new SequenceFile.Reader(fs, p, conf);
        /*
         * new KeyValueLineRecordReader(conf, new FileSplit(p, 0, fs .getFileStatus(p).getLen(), (String[])
         * null));
         */
        Text key = new Text();
        SoftCluster value = new SoftCluster();
        while (converged && reader.next(key, value)) {
          converged = value.isConverged();
        }
      } finally {
        if (reader != null) {
          reader.close();
        }
      }
    }

    return converged;
  }
}
