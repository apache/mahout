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
import java.util.Collection;
import java.util.List;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FuzzyKMeansDriver extends AbstractJob {

  public static final String M_OPTION = "m";

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new FuzzyKMeansDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.clustersInOption()
        .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected"
            + " and written out to this path first")
        .create());
    addOption(DefaultOptionCreator.numClustersOption()
        .withDescription("The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(M_OPTION, M_OPTION, "coefficient normalization factor, must be greater than 1", true);
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

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

    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = ccl.loadClass(measureClass).asSubclass(DistanceMeasure.class).newInstance();

    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(getConf(), input, clusters, Integer.parseInt(parseArguments(args)
          .get(DefaultOptionCreator.NUM_CLUSTERS_OPTION)), measure);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    run(getConf(),
        input,
        clusters,
        output,
        measure,
        convergenceDelta,
        maxIterations,
        fuzziness,
        runClustering,
        emitMostLikely,
        threshold,
        runSequential);
    return 0;
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
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runClustering 
   *          true if points are to be clustered after iterations complete
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void run(Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential) throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut = buildClusters(new Configuration(),
                                     input,
                                     clustersIn,
                                     output,
                                     measure,
                                     convergenceDelta,
                                     maxIterations,
                                     m,
                                     runSequential);
    if (runClustering) {
      log.info("Clustering ");
      clusterData(input,
                  clustersOut,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  measure,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Run the iteration using supplied arguments
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
   * @param m
   *          the fuzzification factor - see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * 
   * @return true if the iteration successfully runs
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private static boolean runIteration(Configuration conf,
                                      Path input,
                                      Path clustersIn,
                                      Path clustersOut,
                                      String measureClass,
                                      double convergenceDelta,
                                      float m) throws IOException, InterruptedException, ClassNotFoundException {

    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    // these values don't matter during iterations as only used for clustering if requested
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(true));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(0));

    Job job = new Job(conf, "FuzzyKMeans Driver running runIteration over clustersIn: " + clustersIn);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(ClusterObservations.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(SoftCluster.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.setMapperClass(FuzzyKMeansMapper.class);
    job.setCombinerClass(FuzzyKMeansCombiner.class);
    job.setReducerClass(FuzzyKMeansReducer.class);
    job.setJarByClass(FuzzyKMeansDriver.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Fuzzy K-Means Iteration failed processing " + clustersIn);
    }
    FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);
    return isConverged(clustersOut, conf, fs);
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runClustering 
   *          true if points are to be clustered after iterations complete
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void run(Configuration conf,
                         Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut =
        buildClusters(conf, input, clustersIn, output, measure, convergenceDelta, maxIterations, m, runSequential);
    if (runClustering) {
      log.info("Clustering");
      clusterData(input,
                  clustersOut,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  measure,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param m
   *          the fuzzification factor, see
   *          http://en.wikipedia.org/wiki/Data_clustering#Fuzzy_c-means_clustering
   * @param runSequential if true run in sequential execution mode
   * 
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf,
                                   Path input,
                                   Path clustersIn,
                                   Path output,
                                   DistanceMeasure measure,
                                   double convergenceDelta,
                                   int maxIterations,
                                   float m,
                                   boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    if (runSequential) {
      return buildClustersSeq(input, clustersIn, output, measure, convergenceDelta, maxIterations, m);
    } else {
      return buildClustersMR(conf, input, clustersIn, output, measure, convergenceDelta, maxIterations, m);
    }
  }

  private static Path buildClustersSeq(Path input,
                                       Path clustersIn,
                                       Path output,
                                       DistanceMeasure measure,
                                       double convergenceDelta,
                                       int maxIterations,
                                       float m) throws IOException {
    FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, convergenceDelta, m);
    List<SoftCluster> clusters = new ArrayList<SoftCluster>();

    FuzzyKMeansUtil.configureWithClusterInfo(clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      log.info("Fuzzy k-Means Iteration: " + iteration);
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(input.toUri(), conf);
      for (VectorWritable value
           : new SequenceFileDirValueIterable<VectorWritable>(input,
                                                              PathType.LIST,
                                                              PathFilters.logsCRCFilter(),
                                                              conf)) {
        clusterer.addPointToClusters(clusters,value.get());
      }
      converged = clusterer.testConvergence(clusters);
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersOut, "part-r-00000"),
                                                           Text.class,
                                                           SoftCluster.class);
      try {
        for (SoftCluster cluster : clusters) {
          log.debug("Writing Cluster:{} center:{} numPoints:{} radius:{} to: {}",
                    new Object[] {
                        cluster.getId(),
                        AbstractCluster.formatVector(cluster.getCenter(), null),
                        cluster.getNumPoints(),
                        AbstractCluster.formatVector(cluster.getRadius(), null),
                        clustersOut.getName()
                    });
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
      clustersIn = clustersOut;
      iteration++;
    }
    return clustersIn;
  }

  private static Path buildClustersMR(Configuration conf,
                                      Path input,
                                      Path clustersIn,
                                      Path output,
                                      DistanceMeasure measure,
                                      double convergenceDelta,
                                      int maxIterations,
                                      float m) throws IOException, InterruptedException, ClassNotFoundException {
    boolean converged = false;
    int iteration = 1;

    // iterate until the clusters converge
    while (!converged && iteration <= maxIterations) {
      log.info("Fuzzy K-Means Iteration {}", iteration);

      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(conf, input, clustersIn, clustersOut, measure.getClass().getName(), convergenceDelta, m);

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
   * @param measure
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void clusterData(Path input,
                                 Path clustersIn,
                                 Path output,
                                 DistanceMeasure measure,
                                 double convergenceDelta,
                                 float m,
                                 boolean emitMostLikely,
                                 double threshold,
                                 boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    if (runSequential) {
      clusterDataSeq(input, clustersIn, output, measure, convergenceDelta, m);
    } else {
      clusterDataMR(input, clustersIn, output, measure, convergenceDelta, m, emitMostLikely, threshold);
    }
  }

  private static void clusterDataSeq(Path input,
                                     Path clustersIn,
                                     Path output,
                                     DistanceMeasure measure,
                                     double convergenceDelta,
                                     float m) throws IOException {
    FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, convergenceDelta, m);
    List<SoftCluster> clusters = new ArrayList<SoftCluster>();
    FuzzyKMeansUtil.configureWithClusterInfo(clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, PathFilters.logsCRCFilter());
    int part = 0;
    for (FileStatus s : status) {
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(output, "part-m-" + part),
                                                           IntWritable.class,
                                                           WeightedVectorWritable.class);
      try {
        for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(s.getPath(), conf)) {
          clusterer.emitPointToClusters(value, clusters, writer);
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
    }

  }

  private static void clusterDataMR(Path input,
                                    Path clustersIn,
                                    Path output,
                                    DistanceMeasure measure,
                                    double convergenceDelta,
                                    float m,
                                    boolean emitMostLikely,
                                    double threshold) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(FuzzyKMeansConfigKeys.M_KEY, String.valueOf(m));
    conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, Double.toString(threshold));

    // Clear output
    output.getFileSystem(conf).delete(output, true);

    Job job = new Job(conf, "FuzzyKMeans Driver running clusterData over input: " + input);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(FuzzyKMeansClusterMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(FuzzyKMeansDriver.class);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Fuzzy K-Means Clustering failed processing " + clustersIn);
    }
  }

  /**
   * Return if all of the Clusters in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   */
  private static boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {

    Path clusterPath = new Path(filePath, "*");
    Collection<Path> result = new ArrayList<Path>();

    FileStatus[] matches =
        fs.listStatus(FileUtil.stat2Paths(fs.globStatus(clusterPath, PathFilters.partFilter())),
                      PathFilters.partFilter());

    for (FileStatus match : matches) {
      result.add(fs.makeQualified(match.getPath()));
    }
    boolean converged = true;

    for (Path path : result) {
      SequenceFileValueIterator<SoftCluster> iterator = new SequenceFileValueIterator<SoftCluster>(path, true, conf);
      try {
        while (converged && iterator.hasNext()) {
          converged = iterator.next().isConverged();
        }
      } finally {
        Closeables.closeQuietly(iterator);
      }
    }

    return converged;
  }
}
