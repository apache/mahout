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

package org.apache.mahout.common.commandline;

import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyDriver;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.kernel.TriangularKernelProfile;

public final class DefaultOptionCreator {
  
  public static final String CLUSTERING_OPTION = "clustering";
  
  public static final String CLUSTERS_IN_OPTION = "clusters";
  
  public static final String CONVERGENCE_DELTA_OPTION = "convergenceDelta";
  
  public static final String DISTANCE_MEASURE_OPTION = "distanceMeasure";
  
  public static final String EMIT_MOST_LIKELY_OPTION = "emitMostLikely";
  
  public static final String INPUT_OPTION = "input";
  
  public static final String MAX_ITERATIONS_OPTION = "maxIter";
  
  public static final String MAX_REDUCERS_OPTION = "maxRed";
  
  public static final String METHOD_OPTION = "method";
  
  public static final String NUM_CLUSTERS_OPTION = "numClusters";
  
  public static final String OUTPUT_OPTION = "output";
  
  public static final String OVERWRITE_OPTION = "overwrite";
  
  public static final String T1_OPTION = "t1";
  
  public static final String T2_OPTION = "t2";
  
  public static final String T3_OPTION = "t3";
  
  public static final String T4_OPTION = "t4";
  
  public static final String THRESHOLD_OPTION = "threshold";
  
  public static final String SEQUENTIAL_METHOD = "sequential";
  
  public static final String MAPREDUCE_METHOD = "mapreduce";
  
  public static final String KERNEL_PROFILE_OPTION = "kernelProfile";
  
  private DefaultOptionCreator() {}
  
  /**
   * Returns a default command line option for help. Used by all clustering jobs
   * and many others
   * */
  public static Option helpOption() {
    return new DefaultOptionBuilder().withLongName("help")
        .withDescription("Print out help").withShortName("h").create();
  }
  
  /**
   * Returns a default command line option for input directory specification.
   * Used by all clustering jobs plus others
   */
  public static DefaultOptionBuilder inputOption() {
    return new DefaultOptionBuilder()
        .withLongName(INPUT_OPTION)
        .withRequired(false)
        .withShortName("i")
        .withArgument(
            new ArgumentBuilder().withName(INPUT_OPTION).withMinimum(1)
                .withMaximum(1).create())
        .withDescription("Path to job input directory.");
  }
  
  /**
   * Returns a default command line option for clusters input directory
   * specification. Used by FuzzyKmeans, Kmeans
   */
  public static DefaultOptionBuilder clustersInOption() {
    return new DefaultOptionBuilder()
        .withLongName(CLUSTERS_IN_OPTION)
        .withRequired(true)
        .withArgument(
            new ArgumentBuilder().withName(CLUSTERS_IN_OPTION).withMinimum(1)
                .withMaximum(1).create())
        .withDescription(
            "The path to the initial clusters directory. Must be a SequenceFile of some type of Cluster")
        .withShortName("c");
  }
  
  /**
   * Returns a default command line option for output directory specification.
   * Used by all clustering jobs plus others
   */
  public static DefaultOptionBuilder outputOption() {
    return new DefaultOptionBuilder()
        .withLongName(OUTPUT_OPTION)
        .withRequired(false)
        .withShortName("o")
        .withArgument(
            new ArgumentBuilder().withName(OUTPUT_OPTION).withMinimum(1)
                .withMaximum(1).create())
        .withDescription("The directory pathname for output.");
  }
  
  /**
   * Returns a default command line option for output directory overwriting.
   * Used by all clustering jobs
   */
  public static DefaultOptionBuilder overwriteOption() {
    return new DefaultOptionBuilder()
        .withLongName(OVERWRITE_OPTION)
        .withRequired(false)
        .withDescription(
            "If present, overwrite the output directory before running job")
        .withShortName("ow");
  }
  
  /**
   * Returns a default command line option for specification of distance measure
   * class to use. Used by Canopy, FuzzyKmeans, Kmeans, MeanShift
   */
  public static DefaultOptionBuilder distanceMeasureOption() {
    return new DefaultOptionBuilder()
        .withLongName(DISTANCE_MEASURE_OPTION)
        .withRequired(false)
        .withShortName("dm")
        .withArgument(
            new ArgumentBuilder().withName(DISTANCE_MEASURE_OPTION)
                .withDefault(SquaredEuclideanDistanceMeasure.class.getName())
                .withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The classname of the DistanceMeasure. Default is SquaredEuclidean");
  }
  
  /**
   * Returns a default command line option for specification of sequential or
   * parallel operation. Used by Canopy, FuzzyKmeans, Kmeans, MeanShift,
   * Dirichlet
   */
  public static DefaultOptionBuilder methodOption() {
    return new DefaultOptionBuilder()
        .withLongName(METHOD_OPTION)
        .withRequired(false)
        .withShortName("xm")
        .withArgument(
            new ArgumentBuilder().withName(METHOD_OPTION)
                .withDefault(MAPREDUCE_METHOD).withMinimum(1).withMaximum(1)
                .create())
        .withDescription(
            "The execution method to use: sequential or mapreduce. Default is mapreduce");
  }
  
  /**
   * Returns a default command line option for specification of T1. Used by
   * Canopy, MeanShift
   */
  public static DefaultOptionBuilder t1Option() {
    return new DefaultOptionBuilder()
        .withLongName(T1_OPTION)
        .withRequired(true)
        .withArgument(
            new ArgumentBuilder().withName(T1_OPTION).withMinimum(1)
                .withMaximum(1).create()).withDescription("T1 threshold value")
        .withShortName(T1_OPTION);
  }
  
  /**
   * Returns a default command line option for specification of T2. Used by
   * Canopy, MeanShift
   */
  public static DefaultOptionBuilder t2Option() {
    return new DefaultOptionBuilder()
        .withLongName(T2_OPTION)
        .withRequired(true)
        .withArgument(
            new ArgumentBuilder().withName(T2_OPTION).withMinimum(1)
                .withMaximum(1).create()).withDescription("T2 threshold value")
        .withShortName(T2_OPTION);
  }
  
  /**
   * Returns a default command line option for specification of T3 (Reducer T1).
   * Used by Canopy, MeanShift
   */
  public static DefaultOptionBuilder t3Option() {
    return new DefaultOptionBuilder()
        .withLongName(T3_OPTION)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(T3_OPTION).withMinimum(1)
                .withMaximum(1).create())
        .withDescription("T3 (Reducer T1) threshold value")
        .withShortName(T3_OPTION);
  }
  
  /**
   * Returns a default command line option for specification of T4 (Reducer T2).
   * Used by Canopy, MeanShift
   */
  public static DefaultOptionBuilder t4Option() {
    return new DefaultOptionBuilder()
        .withLongName(T4_OPTION)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName(T4_OPTION).withMinimum(1)
                .withMaximum(1).create())
        .withDescription("T4 (Reducer T2) threshold value")
        .withShortName(T4_OPTION);
  }
  
  /**
   * Returns a default command line option for specification of max number of
   * iterations. Used by Dirichlet, FuzzyKmeans, Kmeans, LDA
   */
  public static DefaultOptionBuilder maxIterationsOption() {
    // default value used by LDA which overrides withRequired(false)
    return new DefaultOptionBuilder()
        .withLongName(MAX_ITERATIONS_OPTION)
        .withRequired(true)
        .withShortName("x")
        .withArgument(
            new ArgumentBuilder().withName(MAX_ITERATIONS_OPTION)
                .withDefault("-1").withMinimum(1).withMaximum(1).create())
        .withDescription("The maximum number of iterations.");
  }
  
  /**
   * Returns a default command line option for specification of numbers of
   * clusters to create. Used by Dirichlet, FuzzyKmeans, Kmeans
   */
  public static DefaultOptionBuilder numClustersOption() {
    return new DefaultOptionBuilder()
        .withLongName(NUM_CLUSTERS_OPTION)
        .withRequired(false)
        .withArgument(
            new ArgumentBuilder().withName("k").withMinimum(1).withMaximum(1)
                .create()).withDescription("The number of clusters to create")
        .withShortName("k");
  }
  
  /**
   * Returns a default command line option for convergence delta specification.
   * Used by FuzzyKmeans, Kmeans, MeanShift
   */
  public static DefaultOptionBuilder convergenceOption() {
    return new DefaultOptionBuilder()
        .withLongName(CONVERGENCE_DELTA_OPTION)
        .withRequired(false)
        .withShortName("cd")
        .withArgument(
            new ArgumentBuilder().withName(CONVERGENCE_DELTA_OPTION)
                .withDefault("0.5").withMinimum(1).withMaximum(1).create())
        .withDescription("The convergence delta value. Default is 0.5");
  }
  
  /**
   * Returns a default command line option for specifying the max number of
   * reducers. Used by Dirichlet, FuzzyKmeans, Kmeans and LDA
   * 
   * @deprecated
   */
  @Deprecated
  public static DefaultOptionBuilder numReducersOption() {
    return new DefaultOptionBuilder()
        .withLongName(MAX_REDUCERS_OPTION)
        .withRequired(false)
        .withShortName("r")
        .withArgument(
            new ArgumentBuilder().withName(MAX_REDUCERS_OPTION)
                .withDefault("2").withMinimum(1).withMaximum(1).create())
        .withDescription("The number of reduce tasks. Defaults to 2");
  }
  
  /**
   * Returns a default command line option for clustering specification. Used by
   * all clustering except LDA
   */
  public static DefaultOptionBuilder clusteringOption() {
    return new DefaultOptionBuilder()
        .withLongName(CLUSTERING_OPTION)
        .withRequired(false)
        .withDescription(
            "If present, run clustering after the iterations have taken place")
        .withShortName("cl");
  }
  
  /**
   * Returns a default command line option for specifying the emitMostLikely
   * flag. Used by Dirichlet and FuzzyKmeans
   */
  public static DefaultOptionBuilder emitMostLikelyOption() {
    return new DefaultOptionBuilder()
        .withLongName(EMIT_MOST_LIKELY_OPTION)
        .withRequired(false)
        .withShortName("e")
        .withArgument(
            new ArgumentBuilder().withName(EMIT_MOST_LIKELY_OPTION)
                .withDefault("true").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "True if clustering should emit the most likely point only, "
                + "false for threshold clustering. Default is true");
  }
  
  /**
   * Returns a default command line option for specifying the clustering
   * threshold value. Used by Dirichlet and FuzzyKmeans
   */
  public static DefaultOptionBuilder thresholdOption() {
    return new DefaultOptionBuilder()
        .withLongName(THRESHOLD_OPTION)
        .withRequired(false)
        .withShortName("t")
        .withArgument(
            new ArgumentBuilder().withName(THRESHOLD_OPTION).withDefault("0")
                .withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The pdf threshold used for cluster determination. Default is 0");
  }
  
  public static DefaultOptionBuilder kernelProfileOption() {
    return new DefaultOptionBuilder()
        .withLongName(DefaultOptionCreator.KERNEL_PROFILE_OPTION)
        .withRequired(false)
        .withShortName("kp")
        .withArgument(
            new ArgumentBuilder()
                .withName(DefaultOptionCreator.KERNEL_PROFILE_OPTION)
                .withDefault(TriangularKernelProfile.class.getName())
                .withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The classname of the IKernelProfile. Default is TriangularKernelProfile");
  }
  
  public static DefaultOptionBuilder inputIsCanopiesOption() {
    return new DefaultOptionBuilder()
        .withLongName(MeanShiftCanopyDriver.INPUT_IS_CANOPIES_OPTION)
        .withRequired(false)
        .withShortName("ic")
        .withArgument(
            new ArgumentBuilder()
                .withName(MeanShiftCanopyDriver.INPUT_IS_CANOPIES_OPTION)
                .withMinimum(1).withMaximum(1).create())
        .withDescription(
            "If present, the input directory already contains MeanShiftCanopies");
  }
  
}
