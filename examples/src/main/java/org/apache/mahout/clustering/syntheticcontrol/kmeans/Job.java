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

package org.apache.mahout.clustering.syntheticcontrol.kmeans;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.clustering.syntheticcontrol.canopy.InputDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends KMeansDriver {

  private static final Logger log = LoggerFactory.getLogger(Job.class);

  private Job() {
  }

  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      new Job().run(args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      HadoopUtil.overwriteOutput(output);
      new Job().job(new Path("testdata"), output, new EuclideanDistanceMeasure(), 80, 55, 0.5, 10);
    }
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
    addOption(DefaultOptionCreator.numReducersOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());

    Map<String, String> argMap = parseArguments(args);
    if (argMap == null) {
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
    int numReduceTasks = Integer.parseInt(getOption(DefaultOptionCreator.MAX_REDUCERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(measureClass);
    DistanceMeasure measure = (DistanceMeasure) cl.newInstance();
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(input, clusters, Integer.parseInt(argMap
          .get(DefaultOptionCreator.NUM_CLUSTERS_OPTION)), measure);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    runJob(input, clusters, output, measure, convergenceDelta, maxIterations, numReduceTasks, runClustering, false);
    return 0;
  }

  /**
   * Run the kmeans clustering job on an input dataset using the given distance measure, t1, t2 and iteration
   * parameters. All output data will be written to the output directory, which will be initially deleted if
   * it exists. The clustered points will reside in the path <output>/clustered-points. By default, the job
   * expects the a file containing synthetic_control.data as obtained from
   * http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series resides in a directory named
   * "testdata", and writes output to a directory named "output".
   * 
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param measure
   *          the DistanceMeasure to use
   * @param t1
   *          the canopy T1 threshold
   * @param t2
   *          the canopy T2 threshold
   * @param convergenceDelta
   *          the double convergence criteria for iterations
   * @param maxIterations
   *          the int maximum number of iterations
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private void job(Path input,
                   Path output,
                   DistanceMeasure measure,
                   double t1,
                   double t2,
                   double convergenceDelta,
                   int maxIterations) throws IOException, InstantiationException, IllegalAccessException, InterruptedException,
      ClassNotFoundException {
    HadoopUtil.overwriteOutput(output);

    Path directoryContainingConvertedInput = new Path(output, Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT);
    log.info("Preparing Input");
    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
    log.info("Running Canopy to get initial clusters");
    CanopyDriver.runJob(directoryContainingConvertedInput, output, measure, t1, t2, false, false);
    log.info("Running KMeans");
    KMeansDriver.runJob(directoryContainingConvertedInput,
                        new Path(output, Cluster.INITIAL_CLUSTERS_DIR),
                        output,
                        measure,
                        convergenceDelta,
                        maxIterations,
                        1,
                        true,
                        false);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(new Path(output, "clusters-" + maxIterations), new Path(output,
                                                                                                            "clusteredPoints"));
    clusterDumper.printClusters(null);
  }
}
