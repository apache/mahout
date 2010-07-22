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
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FuzzyKMeansDriver extends AbstractJob {

  protected static final String M_OPTION = "m";

  public static final String M_OPTION_KEY = "--" + M_OPTION;

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansDriver.class);

  public FuzzyKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    new FuzzyKMeansDriver().run(args);
  }

  /**
   * Run the job using supplied arguments on a new driver instance (convenience)
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
                            int numReduceTasks,
                            float m,
                            boolean runClustering,
                            boolean emitMostLikely,
                            double threshold) throws IOException, ClassNotFoundException, InterruptedException {

    new FuzzyKMeansDriver().job(input,
                                clustersIn,
                                output,
                                measureClass,
                                convergenceDelta,
                                maxIterations,
                                numReduceTasks,
                                m,
                                runClustering,
                                emitMostLikely,
                                threshold);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.clustersInOption()
        .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected" + " and written out to this path first")
        .create());
    addOption(DefaultOptionCreator.numClustersOption()
        .withDescription("The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(M_OPTION, M_OPTION, "coefficient normalization factor, must be greater than 1", true);
    addOption(DefaultOptionCreator.numReducersOption().create());
    //TODO: addOption(DefaultOptionCreator.numMappersOption().create()); but how to set in new Job?
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path clusters = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    float fuzziness = Float.parseFloat(getOption(M_OPTION));

    int numReduceTasks = Integer.parseInt(getOption(DefaultOptionCreator.MAX_REDUCERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(input, clusters, Integer.parseInt(parseArguments(args)
          .get(DefaultOptionCreator.NUM_CLUSTERS_OPTION)));
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    job(input,
        clusters,
        output,
        measureClass,
        convergenceDelta,
        maxIterations,
        numReduceTasks,
        fuzziness,
        runClustering,
        emitMostLikely,
        threshold);
    return 0;
  }

  /**
   * Run the iteration using supplied arguments
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
   * @param iterationNumber
   *          the iteration number that is going to run
   * @param m
   *          the fuzzification factor - see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @return true if the iteration successfully runs
   * @throws IOException 
   */
  private boolean runIteration(Path input,
                               Path clustersIn,
                               Path clustersOut,
                               String measureClass,
                               double convergenceDelta,
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
    job.setNumReduceTasks(numReduceTasks);
    job.setJarByClass(FuzzyKMeansDriver.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

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
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
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
   * @throws ClassNotFoundException
   * @throws InterruptedException
   */
  public void job(Path input,
                  Path clustersIn,
                  Path output,
                  String measureClass,
                  double convergenceDelta,
                  int maxIterations,
                  int numReduceTasks,
                  float m,
                  boolean runClustering,
                  boolean emitMostLikely,
                  double threshold) throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut = buildClusters(input, clustersIn, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, m);
    if (runClustering) {
      log.info("Clustering ");
      clusterData(input,
                  clustersOut,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  measureClass,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
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
   *          the number of reduce tasks
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @return the Path of the final clusters directory
   * @throws IOException
   */
  public Path buildClusters(Path input,
                            Path clustersIn,
                            Path output,
                            String measureClass,
                            double convergenceDelta,
                            int maxIterations,
                            int numReduceTasks,
                            float m) throws IOException {
    boolean converged = false;
    int iteration = 1;

    // iterate until the clusters converge
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);

      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input, clustersIn, clustersOut, measureClass, convergenceDelta, numReduceTasks, iteration, m);

      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }
    return clustersIn;
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
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @throws IOException 
   */
  public void clusterData(Path input,
                          Path clustersIn,
                          Path output,
                          String measureClass,
                          double convergenceDelta,
                          float m,
                          boolean emitMostLikely,
                          double threshold) throws IOException, ClassNotFoundException, InterruptedException {

    Configuration conf = new Configuration();
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(threshold));

    // Clear output
    output.getFileSystem(conf).delete(output, true);

    Job job = new Job(conf);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(FuzzyKMeansClusterMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(FuzzyKMeansDriver.class);

    job.waitForCompletion(true);
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
  private boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {

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
