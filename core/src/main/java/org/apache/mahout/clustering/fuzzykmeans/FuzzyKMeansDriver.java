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
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassificationDriver;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.FuzzyKMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.topdown.PathDirectory;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
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
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);

    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(getConf(),
                                                 input,
                                                 clusters,
                                                 Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)),
                                                 measure);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    run(getConf(),
        input,
        clusters,
        output,
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
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential) throws IOException, ClassNotFoundException, InterruptedException {
    Configuration conf = new Configuration();
    Path clustersOut = buildClusters(conf,
                                     input,
                                     clustersIn,
                                     output,
                                     convergenceDelta,
                                     maxIterations,
                                     m,
                                     runSequential);
    if (runClustering) {
      log.info("Clustering ");
      clusterData(conf, input,
                  clustersOut,
                  output,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
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
                         double convergenceDelta,
                         int maxIterations,
                         float m,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut =
        buildClusters(conf, input, clustersIn, output, convergenceDelta, maxIterations, m, runSequential);
    if (runClustering) {
      log.info("Clustering");
      clusterData(conf, 
                  input,
                  clustersOut,
                  output,
                  convergenceDelta,
                  m,
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   *
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the file pathname for initial cluster centers
   * @param output
   *          the directory pathname for output points
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
                                   double convergenceDelta,
                                   int maxIterations,
                                   float m,
                                   boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    List<Cluster> clusters = Lists.newArrayList();
    FuzzyKMeansUtil.configureWithClusterInfo(conf, clustersIn, clusters);
    
    if (conf == null) {
      conf = new Configuration();
    }
    
    if (clusters.isEmpty()) {
      throw new IllegalStateException("No input clusters found in " + clustersIn + ". Check your -c argument.");
    }
    
    Path priorClustersPath = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);   
    ClusteringPolicy policy = new FuzzyKMeansClusteringPolicy(m, convergenceDelta);
    ClusterClassifier prior = new ClusterClassifier(clusters, policy);
    prior.writeToSeqFiles(priorClustersPath);
    
    if (runSequential) {
      ClusterIterator.iterateSeq(conf, input, priorClustersPath, output, maxIterations);
    } else {
      ClusterIterator.iterateMR(conf, input, priorClustersPath, output, maxIterations);
    }
    return output;
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
   * @param convergenceDelta
*          the convergence delta value
   * @param emitMostLikely
*          a boolean if true emit only most likely cluster for each point
   * @param threshold
*          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential if true run in sequential execution mode
   */
  public static void clusterData(Configuration conf,
                                 Path input,
                                 Path clustersIn,
                                 Path output,
                                 double convergenceDelta,
                                 float m,
                                 boolean emitMostLikely,
                                 double threshold,
                                 boolean runSequential)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    ClusterClassifier.writePolicy(new FuzzyKMeansClusteringPolicy(m, convergenceDelta), clustersIn);
    ClusterClassificationDriver.run(conf, input, output, new Path(output, PathDirectory.CLUSTERED_POINTS_DIRECTORY),
        threshold, emitMostLikely, runSequential);
  }
}
