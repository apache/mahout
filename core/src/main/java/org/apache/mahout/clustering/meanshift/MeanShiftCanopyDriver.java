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

package org.apache.mahout.clustering.meanshift;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
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
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansConfigKeys;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.common.kernel.IKernelProfile;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;

/**
 * This class implements the driver for Mean Shift Canopy clustering
 * 
 */
public class MeanShiftCanopyDriver extends AbstractJob {

  public static final String MAPRED_REDUCE_TASKS = "mapred.reduce.tasks";

  private static final Logger log = LoggerFactory
      .getLogger(MeanShiftCanopyDriver.class);

  public static final String INPUT_IS_CANOPIES_OPTION = "inputIsCanopies";

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.meanshift.stateInKey";

  private static final String CONTROL_CONVERGED = "control/converged";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new MeanShiftCanopyDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.inputIsCanopiesOption().create());
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.kernelProfileOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    String kernelProfileClass = getOption(DefaultOptionCreator.KERNEL_PROFILE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    double convergenceDelta = Double
        .parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer
        .parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean inputIsCanopies = hasOption(INPUT_IS_CANOPIES_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION)
        .equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD);
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = ccl.loadClass(measureClass).asSubclass(
        DistanceMeasure.class).newInstance();
    IKernelProfile kernelProfile = ccl.loadClass(kernelProfileClass)
        .asSubclass(IKernelProfile.class).newInstance();
    run(getConf(), input, output, measure, kernelProfile, t1, t2,
        convergenceDelta, maxIterations, inputIsCanopies, runClustering,
        runSequential);

    return 0;
  }

  /**
   * Run the job where the input format can be either Vectors or Canopies. If
   * requested, cluster the input data using the computed Canopies
   * 
   * @param conf
   *          the Configuration to use
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param measure
   *          the DistanceMeasure
   * @param kernelProfile
   *          the IKernelProfile
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param maxIterations
   *          an int number of iterations
   * @param inputIsCanopies
   *          true if the input path already contains MeanShiftCanopies and does
   *          not need to be converted from Vectors
   * @param runClustering
   *          true if the input points are to be clustered once the iterations
   *          complete
   * @param runSequential
   *          if true run in sequential execution mode
   */
  public static void run(Configuration conf, Path input, Path output,
      DistanceMeasure measure, IKernelProfile kernelProfile, double t1,
      double t2, double convergenceDelta, int maxIterations,
      boolean inputIsCanopies, boolean runClustering, boolean runSequential)
      throws IOException, InterruptedException, ClassNotFoundException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    if (inputIsCanopies) {
      clustersIn = input;
    } else {
      createCanopyFromVectors(conf, input, clustersIn, measure, runSequential);
    }

    Path clustersOut = buildClusters(conf, clustersIn, output, measure,
        kernelProfile, t1, t2, convergenceDelta, maxIterations, runSequential,
        runClustering);
    if (runClustering) {
      clusterData(inputIsCanopies ? input : new Path(output,
          Cluster.INITIAL_CLUSTERS_DIR), clustersOut, new Path(output,
          Cluster.CLUSTERED_POINTS_DIR), runSequential);
    }
  }

  /**
   * Convert input vectors to MeanShiftCanopies for further processing
   */
  public static void createCanopyFromVectors(Configuration conf, Path input,
      Path output, DistanceMeasure measure, boolean runSequential)
      throws IOException, InterruptedException, ClassNotFoundException {
    if (runSequential) {
      createCanopyFromVectorsSeq(input, output, measure);
    } else {
      createCanopyFromVectorsMR(conf, input, output, measure);
    }
  }

  /**
   * Convert vectors to MeanShiftCanopies sequentially
   * 
   * @param input
   *          the Path to the input VectorWritable data
   * @param output
   *          the Path to the initial clusters directory
   * @param measure
   *          the DistanceMeasure
   */
  private static void createCanopyFromVectorsSeq(Path input, Path output,
      DistanceMeasure measure) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, PathFilters.logsCRCFilter());
    int part = 0;
    int id = 0;
    for (FileStatus s : status) {
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(
          output, "part-m-" + part++), Text.class, MeanShiftCanopy.class);
      try {
        for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(
            s.getPath(), conf)) {
          writer.append(new Text(), MeanShiftCanopy.initialCanopy(value.get(),
              id++, measure));
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
    }
  }

  /**
   * Convert vectors to MeanShiftCanopies using Hadoop
   */
  private static void createCanopyFromVectorsMR(Configuration conf, Path input,
      Path output, DistanceMeasure measure) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass()
        .getName());
    Job job = new Job(conf);
    job.setJarByClass(MeanShiftCanopyDriver.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(MeanShiftCanopy.class);
    job.setMapperClass(MeanShiftCanopyCreatorMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException(
          "Mean Shift createCanopyFromVectorsMR failed on input " + input);
    }
  }

  /**
   * Iterate over the input clusters to produce the next cluster directories for
   * each iteration
   * 
   * @param conf
   *          the Configuration to use
   * @param clustersIn
   *          the input directory Path
   * @param output
   *          the output Path
   * @param measure
   *          the DistanceMeasure
   * @param kernelProfile
   *          the IKernelProfile
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param maxIterations
   *          an int number of iterations
   * @param runSequential
   *          if true run in sequential execution mode
   * @param runClustering
   *          if true accumulate merged clusters for subsequent clustering step
   */
  public static Path buildClusters(Configuration conf, Path clustersIn,
      Path output, DistanceMeasure measure, IKernelProfile kernelProfile,
      double t1, double t2, double convergenceDelta, int maxIterations,
      boolean runSequential, boolean runClustering) throws IOException,
      InterruptedException, ClassNotFoundException {
    if (runSequential) {
      return buildClustersSeq(clustersIn, output, measure, kernelProfile, t1,
          t2, convergenceDelta, maxIterations, runClustering);
    } else {
      return buildClustersMR(conf, clustersIn, output, measure, kernelProfile,
          t1, t2, convergenceDelta, maxIterations, runClustering);
    }
  }

  /**
   * Build new clusters sequentially
   * 
   */
  private static Path buildClustersSeq(Path clustersIn, Path output,
      DistanceMeasure measure, IKernelProfile aKernelProfile, double t1,
      double t2, double convergenceDelta, int maxIterations,
      boolean runClustering) throws IOException {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(measure,
        aKernelProfile, t1, t2, convergenceDelta, runClustering);
    List<MeanShiftCanopy> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(clustersIn.toUri(), conf);
    for (MeanShiftCanopy value : new SequenceFileDirValueIterable<MeanShiftCanopy>(
        clustersIn, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
      clusterer.mergeCanopy(value, clusters);
    }
    boolean[] converged = { false };
    int iteration = 1;
    while (!converged[0] && iteration <= maxIterations) {
      log.info("Mean Shift Iteration: {}", iteration);
      clusters = clusterer.iterate(clusters, converged);
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(
          clustersOut, "part-r-00000"), Text.class, MeanShiftCanopy.class);
      try {
        for (MeanShiftCanopy cluster : clusters) {
          log.debug(
              "Writing Cluster:{} center:{} numPoints:{} radius:{} to: {}",
              new Object[] { cluster.getId(),
                  AbstractCluster.formatVector(cluster.getCenter(), null),
                  cluster.getNumPoints(),
                  AbstractCluster.formatVector(cluster.getRadius(), null),
                  clustersOut.getName() });
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

  /**
   * Build new clusters using Hadoop
   * 
   */
  private static Path buildClustersMR(Configuration conf, Path clustersIn,
      Path output, DistanceMeasure measure, IKernelProfile aKernelProfile,
      double t1, double t2, double convergenceDelta, int maxIterations,
      boolean runClustering) throws IOException, InterruptedException,
      ClassNotFoundException {
    // iterate until the clusters converge
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      int numReducers = Integer.valueOf(conf.get(MAPRED_REDUCE_TASKS, "1"));
      log.info("Mean Shift Iteration: {}, numReducers {}", new Object[] {
          iteration, numReducers });
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      Path controlOut = new Path(output, CONTROL_CONVERGED);
      runIterationMR(conf, clustersIn, clustersOut, controlOut, measure
          .getClass().getName(), aKernelProfile.getClass().getName(), t1, t2,
          convergenceDelta, runClustering);
      converged = FileSystem.get(new Configuration()).exists(controlOut);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
      // decrease the number of reducers if it is > 1 to cross-pollenate
      // map sets
      if (numReducers > 1) {
        numReducers--;
        conf.set(MAPRED_REDUCE_TASKS, String.valueOf(numReducers));
      }
    }
    return clustersIn;
  }

  /**
   * Run an iteration using Hadoop
   * 
   * @param conf
   *          the Configuration to use
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param control
   *          the control path
   * @param measureClassName
   *          the DistanceMeasure class name
   * @param kernelProfileClassName
   *          an IKernel class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param runClustering
   *          if true accumulate merged clusters for subsequent clustering step
   */
  private static void runIterationMR(Configuration conf, Path input,
      Path output, Path control, String measureClassName,
      String kernelProfileClassName, double t1, double t2,
      double convergenceDelta, boolean runClustering) throws IOException,
      InterruptedException, ClassNotFoundException {

    conf.set(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(MeanShiftCanopyConfigKeys.KERNEL_PROFILE_KEY,
        kernelProfileClassName);
    conf.set(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY, String
        .valueOf(convergenceDelta));
    conf.set(MeanShiftCanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(MeanShiftCanopyConfigKeys.T2_KEY, String.valueOf(t2));
    conf.set(MeanShiftCanopyConfigKeys.CONTROL_PATH_KEY, control.toString());
    conf.set(MeanShiftCanopyConfigKeys.CLUSTER_POINTS_KEY, String
        .valueOf(runClustering));
    Job job = new Job(conf,
        "Mean Shift Driver running runIteration over input: " + input);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(MeanShiftCanopy.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(MeanShiftCanopyMapper.class);
    job.setReducerClass(MeanShiftCanopyReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setJarByClass(MeanShiftCanopyDriver.class);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Mean Shift Iteration failed on input "
          + input);
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
   *          the directory pathname for output clustered points
   * @param runSequential
   *          if true run in sequential execution mode
   */
  public static void clusterData(Path input, Path clustersIn, Path output,
      boolean runSequential) throws IOException, InterruptedException,
      ClassNotFoundException {
    if (runSequential) {
      clusterDataSeq(input, clustersIn, output);
    } else {
      clusterDataMR(input, clustersIn, output);
    }
  }

  /**
   * Cluster the data sequentially
   */
  private static void clusterDataSeq(Path input, Path clustersIn, Path output)
      throws IOException {
    Collection<MeanShiftCanopy> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    for (MeanShiftCanopy value : new SequenceFileDirValueIterable<MeanShiftCanopy>(
        clustersIn, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
      clusters.add(value);
    }
    // iterate over all points, assigning each to the closest canopy and
    // outputting that clustering
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, PathFilters.logsCRCFilter());
    int part = 0;
    for (FileStatus s : status) {
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(
          output, "part-m-" + part++), IntWritable.class,
          WeightedVectorWritable.class);
      try {
        for (Pair<Writable, MeanShiftCanopy> record : new SequenceFileIterable<Writable, MeanShiftCanopy>(
            s.getPath(), conf)) {
          MeanShiftCanopy canopy = record.getSecond();
          MeanShiftCanopy closest = MeanShiftCanopyClusterer
              .findCoveringCanopy(canopy, clusters);
          writer.append(new IntWritable(closest.getId()),
              new WeightedVectorWritable(1, canopy.getCenter()));
        }
      } finally {
        Closeables.closeQuietly(writer);
      }
    }
  }

  /**
   * Cluster the data using Hadoop
   */
  private static void clusterDataMR(Path input, Path clustersIn, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, clustersIn.toString());
    Job job = new Job(conf,
        "Mean Shift Driver running clusterData over input: " + input);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);
    job.setMapperClass(MeanShiftCanopyClusterMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(MeanShiftCanopyDriver.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException(
          "Mean Shift Clustering failed on clustersIn " + clustersIn);
    }
  }
}
