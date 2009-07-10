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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.CommandLineUtil;
import org.apache.mahout.utils.HadoopUtil;
import org.apache.mahout.utils.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FuzzyKMeansDriver {

  private static final Logger log = LoggerFactory
      .getLogger(FuzzyKMeansDriver.class);


  private FuzzyKMeansDriver() {
  }


  public static void main(String[] args) throws ClassNotFoundException, IOException, IllegalAccessException, InstantiationException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
        withDescription("The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option clustersOpt = obuilder.withLongName("clusters").withRequired(true).withArgument(
        abuilder.withName("clusters").withMinimum(1).withMaximum(1).create()).
        withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  " +
            "If k is also specified, then a random set of vectors will be selected and written out to this path first").withShortName("c").create();

    Option kOpt = obuilder.withLongName("k").withRequired(false).withArgument(
        abuilder.withName("k").withMinimum(1).withMaximum(1).create()).
        withDescription("The k in k-Means.  If specified, then a random selection of k Vectors will be chosen as the Centroid and written to the clusters output path.").withShortName("k").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
        withDescription("The Path to put the output in").withShortName("o").create();


    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).
        withDescription("The Distance Measure to use.  Default is SquaredEuclidean").withShortName("dm").create();

    Option convergenceDeltaOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(
        abuilder.withName("convergence").withMinimum(1).withMaximum(1).create()).
        withDescription("The threshold below which the clusters are considered to be converged.  Default is 0.5").withShortName("d").create();

    Option maxIterationsOpt = obuilder.withLongName("max").withRequired(false).withArgument(
        abuilder.withName("max").withMinimum(1).withMaximum(1).create()).
        withDescription("The maximum number of iterations to perform.  Default is 20").withShortName("x").create();

    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).
        withDescription("The Vector implementation class name.  Default is SparseVector.class").withShortName("v").create();

    Option helpOpt = obuilder.withLongName("help").
        withDescription("Print out help").withShortName("h").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).
        withDescription("If set, overwrite the output directory").withShortName("w").create();

    Option clusteringOpt = obuilder.withLongName("clustering").withRequired(false).
        withDescription("If true, run clustering only (assumes the iterations have already taken place").withShortName("l").create();

    Option mOpt = obuilder.withLongName("m").withRequired(true).withArgument(
        abuilder.withName("m").withMinimum(1).withMaximum(1).create()).
        withDescription("coefficient normalization factor, must be greater than 1").withShortName("m").create();

    Option numReduceTasksOpt = obuilder.withLongName("numReduce").withRequired(false).withArgument(
        abuilder.withName("numReduce").withMinimum(1).withMaximum(1).create()).
        withDescription("The number of reduce tasks").withShortName("r").create();


    Option numMapTasksOpt = obuilder.withLongName("numMap").withRequired(false).withArgument(
        abuilder.withName("numMap").withMinimum(1).withMaximum(1).create()).
        withDescription("The number of map tasks").withShortName("u").create();


    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(clustersOpt).withOption(outputOpt).withOption(measureClassOpt)
        .withOption(convergenceDeltaOpt).withOption(maxIterationsOpt).withOption(kOpt).withOption(mOpt)
        .withOption(vectorClassOpt).withOption(overwriteOutput).withOption(helpOpt).create();

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
      float m = Float.parseFloat(cmdLine.getValue(mOpt).toString());

      Class<? extends Vector> vectorClass = cmdLine.hasOption(vectorClassOpt) == false ?
          SparseVector.class
          : (Class<? extends Vector>) Class.forName(cmdLine.getValue(vectorClassOpt).toString());


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

      if (cmdLine.hasOption(overwriteOutput) == true) {
        HadoopUtil.overwriteOutput(output);
      }

      if (cmdLine.hasOption(kOpt)) {
        clusters = RandomSeedGenerator.buildRandom(input, clusters,
            Integer.parseInt(cmdLine.getValue(kOpt).toString())).toString();
      }

      if (cmdLine.hasOption(clusteringOpt)) {
        runClustering(input, clusters, output, measureClass, convergenceDelta, numMapTasks, m, vectorClass);
      } else {
        runJob(input, clusters, output, measureClass, convergenceDelta,
            maxIterations, numMapTasks, numReduceTasks, m, vectorClass);
      }


    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }


  }

  /**
   * Run the job using supplied arguments
   *
   * @param input            the directory pathname for input points
   * @param clustersIn       the directory pathname for initial & computed clusters
   * @param output           the directory pathname for output points
   * @param measureClass     the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param maxIterations    the maximum number of iterations
   * @param numMapTasks      the number of mapper tasks
   * @param numReduceTasks   the number of reduce tasks
   * @param m                the fuzzification factor, see http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param vectorClass      the {@link org.apache.mahout.matrix.Vector} implementation to use
   */
  public static void runJob(String input, String clustersIn, String output,
                            String measureClass, double convergenceDelta, int maxIterations,
                            int numMapTasks, int numReduceTasks, float m, Class<? extends Vector> vectorClass) {

    boolean converged = false;
    int iteration = 0;

    // iterate until the clusters converge
    while (!converged && iteration < maxIterations) {
      log.info("Iteration {" + iteration + '}');

      // point the output to a new directory per iteration
      String clustersOut = output + File.separator + "clusters-" + iteration;
      converged = runIteration(input, clustersIn, clustersOut, measureClass,
          convergenceDelta, numMapTasks, numReduceTasks, iteration, m);

      // now point the input to the old output directory
      clustersIn = output + File.separator + "clusters-" + iteration;
      iteration++;
    }

    // now actually cluster the points
    log.info("Clustering ");

    runClustering(input, clustersIn, output + File.separator + "points",
        measureClass, convergenceDelta, numMapTasks, m, vectorClass);
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input            the directory pathname for input points
   * @param clustersIn       the directory pathname for iniput clusters
   * @param clustersOut      the directory pathname for output clusters
   * @param measureClass     the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param numMapTasks      the number of map tasks
   * @param iterationNumber  the iteration number that is going to run
   * @param m                the fuzzification factor - see http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @return true if the iteration successfully runs
   */
  private static boolean runIteration(String input, String clustersIn,
                                      String clustersOut, String measureClass, double convergenceDelta,
                                      int numMapTasks, int numReduceTasks, int iterationNumber, float m) {

    JobConf conf = new JobConf(FuzzyKMeansJob.class);
    conf.setJobName("Fuzzy K Means{" + iterationNumber + '}');

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(FuzzyKMeansInfo.class);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(SoftCluster.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(clustersOut);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.setMapperClass(FuzzyKMeansMapper.class);
    conf.setCombinerClass(FuzzyKMeansCombiner.class);
    conf.setReducerClass(FuzzyKMeansReducer.class);
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(numReduceTasks);

    conf.set(SoftCluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(SoftCluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(SoftCluster.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(SoftCluster.M_KEY, String.valueOf(m));

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");

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
   * @param input            the directory pathname for input points
   * @param clustersIn       the directory pathname for input clusters
   * @param output           the directory pathname for output points
   * @param measureClass     the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param numMapTasks      the number of map tasks
   */
  private static void runClustering(String input, String clustersIn,
                                    String output, String measureClass, double convergenceDelta,
                                    int numMapTasks, float m, Class<? extends Vector> vectorClass) {

    JobConf conf = new JobConf(FuzzyKMeansDriver.class);
    conf.setJobName("Fuzzy K Means Clustering");

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(vectorClass);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(FuzzyKMeansOutput.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(FuzzyKMeansClusterMapper.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");
    conf.setNumMapTasks(numMapTasks);
    conf.setNumReduceTasks(0);
    conf.set(SoftCluster.CLUSTER_PATH_KEY, clustersIn);
    conf.set(SoftCluster.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(SoftCluster.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
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
   * @param conf     the JobConf
   * @param fs       the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException if there was an IO error
   */
  private static boolean isConverged(String filePath, Configuration conf,
                                     FileSystem fs) throws IOException {

    Path clusterPath = new Path(filePath + "/*");
    List<Path> result = new ArrayList<Path>();

    PathFilter clusterFileFilter = new PathFilter() {
      @Override
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

      SequenceFile.Reader reader = null;

      try {
        reader = new SequenceFile.Reader(fs, p, conf);
        /*new KeyValueLineRecordReader(conf, new FileSplit(p, 0, fs
      .getFileStatus(p).getLen(), (String[]) null));*/
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
