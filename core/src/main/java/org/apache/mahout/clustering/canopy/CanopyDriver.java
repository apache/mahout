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

package org.apache.mahout.clustering.canopy;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CanopyDriver extends AbstractJob {

  public static final String DEFAULT_CLUSTERED_POINTS_DIRECTORY = "clusteredPoints";

  private static final Logger log = LoggerFactory.getLogger(CanopyDriver.class);

  public CanopyDriver() {
  }

  public static void main(String[] args) throws Exception {
    new CanopyDriver().run(args);
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
   * @param runClustering 
   *          true if points are to be clustered after clusters are determined
   * @param runSequential execute sequentially if true
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
                            boolean runClustering,
                            boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    new CanopyDriver().job(input, output, measureClassName, t1, t2, runClustering, runSequential);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

    Map<String, String> argMap = parseArguments(args);
    if (argMap == null) {
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
    boolean runSequential = (getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD));

    job(input, output, measureClass, t1, t2, runClustering, runSequential);
    return 0;
  }

  /**
   * Build a directory of Canopy clusters from the input arguments and, if requested,
   * cluster the input vectors using these clusters
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measureClassName the String class name of the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runClustering cluster the input vectors if true
   * @param runSequential execute sequentially if true
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
                  boolean runClustering,
                  boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    Path clustersOut = buildClusters(input, output, measureClassName, t1, t2, runSequential);
    if (runClustering) {
      clusterData(input, clustersOut, output, measureClassName, t1, t2, runSequential);
    }
  }

  /**
   * Build a directory of Canopy clusters from the input vectors and other arguments.
   * Run sequential or mapreduce execution as requested
   * 
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measureClassName the String class name of the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runSequential a boolean indicates to run the sequential (reference) algorithm
   * @return the canopy output directory Path 
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public Path buildClusters(Path input, Path output, String measureClassName, double t1, double t2, boolean runSequential)
      throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    log.info("Input: {} Out: {} " + "Measure: {} t1: {} t2: {}", new Object[] { input, output, measureClassName, t1, t2 });
    if (runSequential) {
      return buildClustersSeq(input, output, measureClassName, t1, t2);
    } else {
      return buildClustersMR(input, output, measureClassName, t1, t2);
    }
  }

  /**
   * Build a directory of Canopy clusters from the input vectors and other arguments.
   * Run sequential execution
   * 
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measureClassName the String class name of the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runSequential a boolean indicates to run sequential (reference) algorithm
   * @return the canopy output directory Path 
   * @throws ClassNotFoundException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IOException 
   */
  private Path buildClustersSeq(Path input, Path output, String measureClassName, double t1, double t2)
      throws InstantiationException, IllegalAccessException, ClassNotFoundException, IOException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = (DistanceMeasure) ccl.loadClass(measureClassName).newInstance();
    CanopyClusterer clusterer = new CanopyClusterer(measure, t1, t2);
    List<Canopy> canopies = new ArrayList<Canopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
        VectorWritable vw = (VectorWritable) reader.getValueClass().newInstance();
        while (reader.next(key, vw)) {
          clusterer.addPointToCanopies(vw.get(), canopies);
          vw = (VectorWritable) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
      }
    }
    Path canopyOutputDir = new Path(output, Cluster.CLUSTERS_DIR + '0');
    Path path = new Path(canopyOutputDir, "part-r-00000");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Canopy.class);
    try {
      for (Canopy canopy : canopies) {
        log.info("Writing Canopy:" + canopy.getIdentifier() + " center:" + ClusterBase.formatVector(canopy.getCenter(), null)
            + " numPoints:" + canopy.getNumPoints() + " centroid:" + ClusterBase.formatVector(canopy.computeCentroid(), null));
        writer.append(new Text(canopy.getIdentifier()), canopy);
      }
    } finally {
      writer.close();
    }
    return canopyOutputDir;
  }

  /**
   * Build a directory of Canopy clusters from the input vectors and other arguments.
   * Run mapreduce execution
   * 
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measureClassName the String class name of the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runSequential a boolean indicates to run sequential (reference) algorithm
   * @return the canopy output directory Path 
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private Path buildClustersMR(Path input, Path output, String measureClassName, double t1, double t2) throws IOException,
      InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(t2));

    Job job = new Job(conf);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(CanopyMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setReducerClass(CanopyReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Canopy.class);
    job.setNumReduceTasks(1);
    job.setJarByClass(CanopyDriver.class);

    FileInputFormat.addInputPath(job, input);
    Path canopyOutputDir = new Path(output, Cluster.CLUSTERS_DIR + '0');
    FileOutputFormat.setOutputPath(job, canopyOutputDir);
    job.waitForCompletion(true);
    return canopyOutputDir;
  }

  /**
   * Cluster the points using the given canopies and other arguments
   * 
   * @param points the input points directory pathname String
   * @param canopies the input canopies directory pathname String
   * @param output the output directory pathname String
   * @param measureClassName the DistanceMeasure class name
   * @param t1 the T1 distance threshold
   * @param t2 the T2 distance threshold
   * @param runSequential a boolean indicates to run the sequential (reference) algorithm
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void clusterData(Path points,
                          Path canopies,
                          Path output,
                          String measureClassName,
                          double t1,
                          double t2,
                          boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    if (runSequential) {
      clusterDataSeq(points, canopies, output, measureClassName, t1, t2);
    } else {
      clusterDataMR(points, canopies, output, measureClassName, t1, t2);
    }
  }

  private void clusterDataSeq(Path points, Path canopies, Path output, String measureClassName, double t1, double t2)
      throws InstantiationException, IllegalAccessException, ClassNotFoundException, IOException, InterruptedException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = (DistanceMeasure) ccl.loadClass(measureClassName).newInstance();
    CanopyClusterer clusterer = new CanopyClusterer(measure, t1, t2);

    List<Canopy> clusters = new ArrayList<Canopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(canopies.toUri(), conf);
    FileStatus[] status = fs.listStatus(canopies, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Text key = (Text) reader.getKeyClass().newInstance();
        Canopy value = (Canopy) reader.getValueClass().newInstance();
        while (reader.next(key, value)) {
          clusters.add(value);
          value = (Canopy) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
      }
    }
    // iterate over all points, assigning each to the closest canopy and outputing that clustering
    fs = FileSystem.get(points.toUri(), conf);
    status = fs.listStatus(points, new OutputLogFilter());
    Path outPath = new Path(output, DEFAULT_CLUSTERED_POINTS_DIRECTORY);
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
        VectorWritable vw = (VectorWritable) reader.getValueClass().newInstance();
        while (reader.next(key, vw)) {
          Canopy closest = clusterer.findClosestCanopy(vw.get(), clusters);
          writer.append(new IntWritable(closest.getId()), new WeightedVectorWritable(1, vw));
          vw = (VectorWritable) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }
  }

  /**
   * @param points
   * @param canopies
   * @param output
   * @param measureClassName
   * @param t1
   * @param t2
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private void clusterDataMR(Path points, Path canopies, Path output, String measureClassName, double t1, double t2)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(t2));
    conf.set(CanopyConfigKeys.CANOPY_PATH_KEY, canopies.toString());

    Job job = new Job(conf);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(ClusterMapper.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(CanopyDriver.class);

    FileInputFormat.addInputPath(job, points);
    Path outPath = new Path(output, DEFAULT_CLUSTERED_POINTS_DIRECTORY);
    FileOutputFormat.setOutputPath(job, outPath);
    HadoopUtil.overwriteOutput(outPath);

    job.waitForCompletion(true);
  }

}
