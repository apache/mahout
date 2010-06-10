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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FuzzyKMeansDriver {

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansDriver.class);

  private FuzzyKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option measureClassOpt = DefaultOptionCreator.distanceMeasureOption().create();
    Option clustersOpt = DefaultOptionCreator.clustersInOption().withDescription(
        "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected"
            + " and written out to this path first")
        .create();
    Option kOpt = DefaultOptionCreator.kOption().withDescription(
        "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption().create();
    Option maxIterationsOpt = DefaultOptionCreator.maxIterationsOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option overwriteOutput = DefaultOptionCreator.overwriteOption().create();
    Option mOpt = DefaultOptionCreator.mOption().create();
    Option numReduceTasksOpt = DefaultOptionCreator.numReducersOption().create();
    Option numMapTasksOpt = DefaultOptionCreator.numMappersOption().create();
    Option clusteringOpt = DefaultOptionCreator.clusteringOption().create();
    Option emitMostLikelyOpt = DefaultOptionCreator.emitMostLikelyOption().create();
    Option thresholdOpt = DefaultOptionCreator.thresholdOption().create();

    Group group = new GroupBuilder().withName("Options").withOption(inputOpt).withOption(clustersOpt)
        .withOption(outputOpt).withOption(measureClassOpt).withOption(convergenceDeltaOpt)
        .withOption(maxIterationsOpt).withOption(kOpt).withOption(mOpt)
        .withOption(overwriteOutput).withOption(helpOpt).withOption(numMapTasksOpt)
        .withOption(numReduceTasksOpt).withOption(emitMostLikelyOpt).withOption(thresholdOpt).create();

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
      String measureClass = SquaredEuclideanDistanceMeasure.class.getName();
      if (cmdLine.hasOption(measureClassOpt)) {
        measureClass = cmdLine.getValue(measureClassOpt).toString();
      }
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      float m = Float.parseFloat(cmdLine.getValue(mOpt).toString());

      int numReduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      int numMapTasks = Integer.parseInt(cmdLine.getValue(numMapTasksOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt).toString());
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      boolean emitMostLikely = Boolean.parseBoolean(cmdLine.getValue(emitMostLikelyOpt).toString());
      double threshold = Double.parseDouble(cmdLine.getValue(thresholdOpt).toString());
      if (cmdLine.hasOption(kOpt)) {
        clusters = RandomSeedGenerator.buildRandom(input, clusters,
                                                   Integer.parseInt(cmdLine.getValue(kOpt).toString()));
      }
      runJob(input,
             clusters,
             output,
             measureClass,
             convergenceDelta,
             maxIterations,
             numMapTasks,
             numReduceTasks,
             m,
             cmdLine.hasOption(clusteringOpt),
             emitMostLikely,
             threshold);

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
   * @throws IOException 
   */
  public static void runJob(Path input,
                            Path clustersIn,
                            Path output,
                            String measureClass,
                            double convergenceDelta,
                            int maxIterations,
                            int numMapTasks,
                            int numReduceTasks,
                            float m,
                            boolean runClustering,
                            boolean emitMostLikely,
                            double threshold) throws IOException {

    boolean converged = false;
    int iteration = 1;

    // iterate until the clusters converge
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);

      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input,
                               clustersIn,
                               clustersOut,
                               measureClass,
                               convergenceDelta,
                               numMapTasks,
                               numReduceTasks,
                               iteration,
                               m);

      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }

    // now actually cluster the points
    log.info("Clustering ");
    runClustering(input,
                  clustersIn,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  measureClass,
                  convergenceDelta,
                  numMapTasks,
                  m,
                  emitMostLikely,
                  threshold);
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
   * @throws IOException 
   */
  private static boolean runIteration(Path input,
                                      Path clustersIn,
                                      Path clustersOut,
                                      String measureClass,
                                      double convergenceDelta,
                                      int numMapTasks,
                                      int numReduceTasks,
                                      int iterationNumber,
                                      float m) throws IOException {

    Configuration conf = new Configuration();
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    // these values don't matter during iterations as only used for clustering if requested
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(true));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(0));
    
    Job job = new Job(conf);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(FuzzyKMeansInfo.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(SoftCluster.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.setMapperClass(FuzzyKMeansMapper.class);
    job.setCombinerClass(FuzzyKMeansCombiner.class);
    job.setReducerClass(FuzzyKMeansReducer.class);
    //TODO: job.setNumMapTasks(numMapTasks);
    job.setNumReduceTasks(numReduceTasks);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");

    try {
      job.waitForCompletion(true);
      FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);
      return isConverged(clustersOut, conf, fs);
    } catch (IOException e) {
      log.warn(e.toString(), e);
      return true;
    } catch (InterruptedException e) {
      log.warn(e.toString(), e);
      return true;
    } catch (ClassNotFoundException e) {
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
   * @throws IOException 
   */
  private static void runClustering(Path input,
                                    Path clustersIn,
                                    Path output,
                                    String measureClass,
                                    double convergenceDelta,
                                    int numMapTasks,
                                    float m,
                                    boolean emitMostLikely,
                                    double threshold) throws IOException {

    Configuration conf = new Configuration();
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(threshold));
    
    Job job = new Job(conf);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(FuzzyKMeansClusterMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    //TODO: job.setNumMapTasks(numMapTasks);
    job.setNumReduceTasks(0);
    try {
      job.waitForCompletion(true);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
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

    FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(
        fs.globStatus(clusterPath, clusterFileFilter)), clusterFileFilter);

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
