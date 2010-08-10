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
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MeanShiftCanopyDriver extends AbstractJob {

  protected static final String INPUT_IS_CANOPIES_OPTION = "inputIsCanopies";

  private static final Logger log = LoggerFactory.getLogger(MeanShiftCanopyDriver.class);

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.meanshift.stateInKey";

  private static final String CONTROL_CONVERGED = "control/converged";

  public static void main(String[] args) throws Exception {
    new MeanShiftCanopyDriver().run(args);
  }

  /**
   * Run the job on a new driver instance (convenience)
   * 
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param measureClassName
   *          the DistanceMeasure class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param maxIterations
   *          an int number of iterations
   * @param inputIsCanopies 
              true if the input path already contains MeanShiftCanopies and does not need to be converted from Vectors
   * @param runClustering 
   *          true if the input points are to be clustered once the iterations complete
   * @param runSequential if true run in sequential execution mode
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public static void runJob(Path input,
                            Path output,
                            String measureClassName,
                            double t1,
                            double t2,
                            double convergenceDelta,
                            int maxIterations,
                            boolean inputIsCanopies,
                            boolean runClustering,
                            boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    new MeanShiftCanopyDriver().job(input,
                                    output,
                                    measureClassName,
                                    t1,
                                    t2,
                                    convergenceDelta,
                                    maxIterations,
                                    inputIsCanopies,
                                    runClustering,
                                    runSequential);
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
   */
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(INPUT_IS_CANOPIES_OPTION,
              INPUT_IS_CANOPIES_OPTION,
              "If present, the input directory already contains MeanShiftCanopies");
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
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
      HadoopUtil.overwriteOutput(output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean inputIsCanopies = hasOption(INPUT_IS_CANOPIES_OPTION);
    boolean runSequential = (getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD));

    job(input, output, measureClass, t1, t2, convergenceDelta, maxIterations, inputIsCanopies, runClustering, runSequential);
    return 0;
  }

  /**
   * Run an iteration
   * 
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param control
   *          the control path
   * @param measureClassName
   *          the DistanceMeasure class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @throws IOException 
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private static void runIteration(Path input,
                                   Path output,
                                   Path control,
                                   String measureClassName,
                                   double t1,
                                   double t2,
                                   double convergenceDelta)
      throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    conf.set(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY, String.valueOf(convergenceDelta));
    conf.set(MeanShiftCanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(MeanShiftCanopyConfigKeys.T2_KEY, String.valueOf(t2));
    conf.set(MeanShiftCanopyConfigKeys.CONTROL_PATH_KEY, control.toString());

    Job job = new Job(conf);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(MeanShiftCanopy.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(MeanShiftCanopyMapper.class);
    job.setReducerClass(MeanShiftCanopyReducer.class);
    job.setNumReduceTasks(1);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setJarByClass(MeanShiftCanopyDriver.class);
    job.waitForCompletion(true);
  }

  /**
   * Run the job where the input format can be either Vectors or Canopies.
   * If requested, cluster the input data using the computed Canopies
   * 
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param measureClassName
   *          the DistanceMeasure class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param maxIterations
   *          an int number of iterations
   * @param inputIsCanopies 
              true if the input path already contains MeanShiftCanopies and does not need to be converted from Vectors
   * @param runClustering 
   *          true if the input points are to be clustered once the iterations complete
   * @param runSequential if true run in sequential execution mode
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void job(Path input,
                  Path output,
                  String measureClassName,
                  double t1,
                  double t2,
                  double convergenceDelta,
                  int maxIterations,
                  boolean inputIsCanopies,
                  boolean runClustering,
                  boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    if (inputIsCanopies) {
      clustersIn = input;
    } else {
      createCanopyFromVectors(input, clustersIn, runSequential);
    }
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(measureClassName);
    DistanceMeasure measure = (DistanceMeasure) cl.newInstance();

    Path clustersOut = buildClusters(clustersIn, output, measure, t1, t2, convergenceDelta, maxIterations, runSequential);
    if (runClustering) {
      clusterData(inputIsCanopies ? input : new Path(output, Cluster.INITIAL_CLUSTERS_DIR),
                  clustersOut,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  runSequential,
                  measure);
    }
  }

  /**
   * @param input
   * @param output
   * @param runSequential
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void createCanopyFromVectors(Path input, Path output, boolean runSequential) throws IOException, InterruptedException,
      ClassNotFoundException, InstantiationException, IllegalAccessException {
    if (runSequential) {
      createCanopyFromVectorsSeq(input, output);
    } else {
      createCanopyFromVectorsMR(input, output);
    }
  }

  /**
   * @param input the Path to the input VectorWritable data
   * @param output the Path to the initial clusters directory
   */
  private static void createCanopyFromVectorsSeq(Path input, Path output)
      throws IOException, InstantiationException, IllegalAccessException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int part = 0;
    int id = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(output, "part-m-" + part++),
                                                           Text.class,
                                                           MeanShiftCanopy.class);
      try {
        WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
        VectorWritable vw = (VectorWritable) reader.getValueClass().newInstance();
        while (reader.next(key, vw)) {
          writer.append(new Text(), new MeanShiftCanopy(vw.get(), id++));
          vw = (VectorWritable) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }
  }

  private static void createCanopyFromVectorsMR(Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
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

    job.waitForCompletion(true);
  }

  /**
   * Iterate over the input clusters to produce the next cluster directories for each iteration
   * 
   * @param clustersIn
   *          the input directory Path
   * @param output
   *          the output Path
   * @param measure
   *          the DistanceMeasure class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @param convergenceDelta
   *          the double convergence criteria
   * @param maxIterations
   *          an int number of iterations
   * @param runSequential if true run in sequential execution mode
   */
  public Path buildClusters(Path clustersIn,
                            Path output,
                            DistanceMeasure measure,
                            double t1,
                            double t2,
                            double convergenceDelta,
                            int maxIterations,
                            boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    if (runSequential) {
      return buildClustersSeq(clustersIn, output, measure, t1, t2, convergenceDelta, maxIterations);
    } else {
      return buildClustersMR(clustersIn, output, measure, t1, t2, convergenceDelta, maxIterations);
    }
  }

  private static Path buildClustersSeq(Path clustersIn,
                                       Path output,
                                       DistanceMeasure measure,
                                       double t1,
                                       double t2,
                                       double convergenceDelta,
                                       int maxIterations)
      throws IOException, InstantiationException, IllegalAccessException {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(measure, t1, t2, convergenceDelta);
    List<MeanShiftCanopy> clusters = new ArrayList<MeanShiftCanopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(clustersIn.toUri(), conf);
    FileStatus[] status = fs.listStatus(clustersIn, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
        MeanShiftCanopy canopy = (MeanShiftCanopy) reader.getValueClass().newInstance();
        while (reader.next(key, canopy)) {
          clusterer.mergeCanopy(canopy, clusters);
          canopy = (MeanShiftCanopy) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
      }
    }
    boolean[] converged = { false };
    int iteration = 1;
    while (!converged[0] && iteration <= maxIterations) {
      log.info("Iteration: " + iteration);
      clusters = clusterer.iterate(clusters, converged);
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersOut, "part-r-00000"),
                                                           Text.class,
                                                           MeanShiftCanopy.class);
      try {
        for (MeanShiftCanopy cluster : clusters) {
          log.info("Writing Cluster:" + cluster.getId() + " center:" + AbstractCluster.formatVector(cluster.getCenter(), null)
                   + " numPoints:" + cluster.getNumPoints() + " radius:" + AbstractCluster.formatVector(cluster.getRadius(), null) + " to: "
                   + clustersOut.getName());
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        writer.close();
      }
      clustersIn = clustersOut;
      iteration++;
    }
    return clustersIn;
  }

  /**
   * @param clustersIn
   * @param output
   * @param measure
   * @param t1
   * @param t2
   * @param convergenceDelta
   * @param maxIterations
   * @return
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private Path buildClustersMR(Path clustersIn,
                               Path output,
                               DistanceMeasure measure,
                               double t1,
                               double t2,
                               double convergenceDelta,
                               int maxIterations) throws IOException, InterruptedException, ClassNotFoundException {
    // iterate until the clusters converge
    boolean converged = false;
    int iteration = 1;
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      Path controlOut = new Path(output, CONTROL_CONVERGED);
      runIteration(clustersIn, clustersOut, controlOut, measure.getClass().getName(), t1, t2, convergenceDelta);
      converged = FileSystem.get(new Configuration()).exists(controlOut);
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
   *          the directory pathname for output clustered points
   * @param runSequential if true run in sequential execution mode
   * @param measure the DistanceMeasure to use
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IOException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void clusterData(Path input, Path clustersIn, Path output, boolean runSequential, DistanceMeasure measure)
      throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    if (runSequential) {
      clusterDataSeq(input, clustersIn, output, measure);
    } else {
      clusterDataMR(input, clustersIn, output);
    }
  }

  private static void clusterDataSeq(Path input, Path clustersIn, Path output, DistanceMeasure measure)
      throws IOException, InstantiationException, IllegalAccessException {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(measure, 0, 0, 0);

    List<MeanShiftCanopy> clusters = new ArrayList<MeanShiftCanopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(clustersIn.toUri(), conf);
    FileStatus[] status = fs.listStatus(clustersIn, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Text key = (Text) reader.getKeyClass().newInstance();
        MeanShiftCanopy value = (MeanShiftCanopy) reader.getValueClass().newInstance();
        while (reader.next(key, value)) {
          clusters.add(value);
          value = (MeanShiftCanopy) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
      }
    }
    // iterate over all points, assigning each to the closest canopy and outputting that clustering
    fs = FileSystem.get(input.toUri(), conf);
    status = fs.listStatus(input, new OutputLogFilter());
    Path outPath = new Path(output, CanopyDriver.DEFAULT_CLUSTERED_POINTS_DIRECTORY);
    int part = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(outPath, "part-m-" + part++),
                                                           IntWritable.class,
                                                           WeightedVectorWritable.class);
      try {
        WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
        MeanShiftCanopy canopy = (MeanShiftCanopy) reader.getValueClass().newInstance();
        while (reader.next(key, canopy)) {
          MeanShiftCanopy closest = clusterer.findCoveringCanopy(canopy, clusters);
          writer.append(new IntWritable(closest.getId()), new WeightedVectorWritable(1, new VectorWritable(canopy.getCenter())));
          canopy = (MeanShiftCanopy) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }
  }

  private static void clusterDataMR(Path input, Path clustersIn, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, clustersIn.toString());
    Job job = new Job(conf);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);
    job.setMapperClass(MeanShiftCanopyClusterMapper.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(MeanShiftCanopyDriver.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.waitForCompletion(true);
  }
}
