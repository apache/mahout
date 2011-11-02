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
import java.util.Collection;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
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
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.clustering.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
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

public class KMeansDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new KMeansDriver(), args);
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
    addOption(DefaultOptionCreator.clusteringOption().create());
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
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);

    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(getConf(), input, clusters, Integer
          .parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)), measure);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    if (getConf() == null) {
      setConf(new Configuration());
    }
    run(getConf(), input, clusters, output, measure, convergenceDelta, maxIterations, runClustering, runSequential);
    return 0;
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
   * @param measure 
   *          the DistanceMeasure to use
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @param runSequential if true execute sequential algorithm
   */
  public static void run(Configuration conf,
                         Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         boolean runClustering,
                         boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {

    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}",
               new Object[] {input, clustersIn, output,measure.getClass().getName()});
      log.info("convergence: {} max Iterations: {} num Reduce Tasks: {} Input Vectors: {}",
               new Object[] {convergenceDelta, maxIterations, VectorWritable.class.getName()});
    }
    Path clustersOut = buildClusters(conf, input, clustersIn, output, measure, maxIterations, delta, runSequential);
    if (runClustering) {
      log.info("Clustering data");
      clusterData(conf,
          input,
          clustersOut,
          new Path(output, AbstractCluster.CLUSTERED_POINTS_DIR),
          measure,
          delta,
          runSequential);
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
   * @param measure 
   *          the DistanceMeasure to use
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @param runSequential if true execute sequential algorithm
   */
  public static void run(Path input,
                         Path clustersIn,
                         Path output,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations,
                         boolean runClustering,
                         boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    run(new Configuration(),
        input,
        clustersIn,
        output,
        measure,
        convergenceDelta,
        maxIterations,
        runClustering,
        runSequential);
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * @param conf 
   *          the Configuration to use
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param maxIterations
   *          the maximum number of iterations
   * @param delta
   *          the convergence delta value
   * @param runSequential if true execute sequential algorithm
   * 
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf,
                                   Path input,
                                   Path clustersIn,
                                   Path output,
                                   DistanceMeasure measure,
                                   int maxIterations,
                                   String delta,
                                   boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    if (runSequential) {
      return buildClustersSeq(conf, input, clustersIn, output, measure, maxIterations, delta);
    } else {
      return buildClustersMR(conf, input, clustersIn, output, measure, maxIterations, delta);
    }
  }

  private static Path buildClustersSeq(Configuration conf,
                                       Path input,
                                       Path clustersIn,
                                       Path output,
                                       DistanceMeasure measure,
                                       int maxIterations,
                                       String delta)
    throws IOException {

    KMeansClusterer clusterer = new KMeansClusterer(measure);
    Collection<Cluster> clusters = Lists.newArrayList();

    KMeansUtil.configureWithClusterInfo(conf, clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      log.info("K-Means Iteration: {}", iteration);
      FileSystem fs = FileSystem.get(input.toUri(), conf);
      for (VectorWritable value
           : new SequenceFileDirValueIterable<VectorWritable>(input,
                                                              PathType.LIST,
                                                              PathFilters.logsCRCFilter(),
                                                              conf)) {
        clusterer.addPointToNearestCluster(value.get(), clusters);
      }
      converged = clusterer.testConvergence(clusters, Double.parseDouble(delta));
      Path clustersOut = new Path(output, AbstractCluster.CLUSTERS_DIR + iteration);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersOut, "part-r-00000"),
                                                           Text.class,
                                                           Cluster.class);
      try {
        for (Cluster cluster : clusters) {
          log.debug("Writing Cluster:{} center:{} numPoints:{} radius:{} to: {}",
                    new Object[] {
                        cluster.getId(),
                        AbstractCluster.formatVector(cluster.getCenter(), null),
                        cluster.getNumPoints(),
                        AbstractCluster.formatVector(cluster.getRadius(), null), clustersOut.getName()
                    });
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
      clustersIn = clustersOut;
      iteration++;
    }
    Path finalClustersIn = new Path(output, AbstractCluster.CLUSTERS_DIR + (iteration-1) + org.apache.mahout.clustering.Cluster.FINAL_ITERATION_SUFFIX);
    FileSystem.get(conf).rename(new Path(output, AbstractCluster.CLUSTERS_DIR + (iteration-1)), finalClustersIn);
    return finalClustersIn;
  }

  private static Path buildClustersMR(Configuration conf,
                                      Path input,
                                      Path clustersIn,
                                      Path output,
                                      DistanceMeasure measure,
                                      int maxIterations,
                                      String delta) throws IOException, InterruptedException, ClassNotFoundException {

    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      log.info("K-Means Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, AbstractCluster.CLUSTERS_DIR + iteration);
      converged = runIteration(conf, input, clustersIn, clustersOut, measure.getClass().getName(), delta);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }
    Path finalClustersIn = new Path(output, AbstractCluster.CLUSTERS_DIR + (iteration-1) + "-final");
    FileSystem.get(conf).rename(new Path(output, AbstractCluster.CLUSTERS_DIR + (iteration-1)), finalClustersIn);
    return finalClustersIn;
  }

  /**
   * Run the job using supplied arguments
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
   * 
   * @return true if the iteration successfully runs
   */
  private static boolean runIteration(Configuration conf,
                                      Path input,
                                      Path clustersIn,
                                      Path clustersOut,
                                      String measureClass,
                                      String convergenceDelta)
    throws IOException, InterruptedException, ClassNotFoundException {

    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf, "KMeans Driver running runIteration over clustersIn: " + clustersIn);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(ClusterObservations.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Cluster.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(KMeansMapper.class);
    job.setCombinerClass(KMeansCombiner.class);
    job.setReducerClass(KMeansReducer.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    job.setJarByClass(KMeansDriver.class);
    HadoopUtil.delete(conf, clustersOut);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("K-Means Iteration failed processing " + clustersIn);
    }
    FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);

    return isConverged(clustersOut, conf, fs);
  }

  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   */
  private static boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {
    for (FileStatus part : fs.listStatus(filePath, PathFilters.partFilter())) {
      SequenceFileValueIterator<Cluster> iterator = new SequenceFileValueIterator<Cluster>(part.getPath(), true, conf);
      while (iterator.hasNext()) {
        Cluster value = iterator.next();
        if (!value.isConverged()) {
          Closeables.closeQuietly(iterator);
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Run the job using supplied arguments
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
   * @param runSequential if true execute sequential algorithm
   */
  public static void clusterData(Configuration conf,
                                 Path input,
                                 Path clustersIn,
                                 Path output,
                                 DistanceMeasure measure,
                                 String convergenceDelta,
                                 boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {

    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}", new Object[] {input, clustersIn, output, measure});
      log.info("convergence: {} Input Vectors: {}", convergenceDelta, VectorWritable.class.getName());
    }
    if (runSequential) {
      clusterDataSeq(conf, input, clustersIn, output, measure);
    } else {
      clusterDataMR(conf, input, clustersIn, output, measure, convergenceDelta);
    }
  }

  private static void clusterDataSeq(Configuration conf,
                                     Path input,
                                     Path clustersIn,
                                     Path output,
                                     DistanceMeasure measure) throws IOException {

    KMeansClusterer clusterer = new KMeansClusterer(measure);
    Collection<Cluster> clusters = Lists.newArrayList();
    KMeansUtil.configureWithClusterInfo(conf, clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
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
          clusterer.emitPointToNearestCluster(value.get(), clusters, writer);
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
    }

  }

  private static void clusterDataMR(Configuration conf,
                                    Path input,
                                    Path clustersIn,
                                    Path output,
                                    DistanceMeasure measure,
                                    String convergenceDelta)
    throws IOException, InterruptedException, ClassNotFoundException {

    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf, "KMeans Driver running clusterData over input: " + input);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedPropertyVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    HadoopUtil.delete(conf, output);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(KMeansClusterMapper.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(KMeansDriver.class);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("K-Means Clustering failed processing " + clustersIn);
    }
  }
}
