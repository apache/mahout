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
import java.util.Collection;
import java.util.List;
import java.util.Map;

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

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CanopyDriver(), args);
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
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = ccl.loadClass(measureClass).asSubclass(DistanceMeasure.class).newInstance();

    run(getConf(), input, output, measure, t1, t2, runClustering, runSequential);
    return 0;
  }

  /**
   * Build a directory of Canopy clusters from the input arguments and, if requested,
   * cluster the input vectors using these clusters
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runClustering cluster the input vectors if true
   * @param runSequential execute sequentially if true
   */
  public static void run(Configuration conf,
                         Path input,
                         Path output,
                         DistanceMeasure measure,
                         double t1,
                         double t2,
                         boolean runClustering,
                         boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    Path clustersOut = buildClusters(conf, input, output, measure, t1, t2, runSequential);
    if (runClustering) {
      clusterData(conf, input, clustersOut, output, measure, t1, t2, runSequential);
    }
  }

  /**
   * Convenience method creates new Configuration()
   * Build a directory of Canopy clusters from the input arguments and, if requested,
   * cluster the input vectors using these clusters
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runClustering cluster the input vectors if true
   * @param runSequential execute sequentially if true
   */
  public static void run(Path input,
                         Path output,
                         DistanceMeasure measure,
                         double t1,
                         double t2,
                         boolean runClustering,
                         boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    run(new Configuration(), input, output, measure, t1, t2, runClustering, runSequential);
  }

  /**
   * Build a directory of Canopy clusters from the input vectors and other arguments.
   * Run sequential or mapreduce execution as requested
   * 
   * @param conf the Configuration to use
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measure the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @param runSequential a boolean indicates to run the sequential (reference) algorithm
   * @return the canopy output directory Path
   */
  public static Path buildClusters(Configuration conf,
                                   Path input,
                                   Path output,
                                   DistanceMeasure measure,
                                   double t1,
                                   double t2,
                                   boolean runSequential) throws InstantiationException, IllegalAccessException, IOException,
      InterruptedException, ClassNotFoundException {
    log.info("Input: {} Out: {} " + "Measure: {} t1: {} t2: {}", new Object[] { input, output, measure, t1, t2 });
    if (runSequential) {
      return buildClustersSeq(input, output, measure, t1, t2);
    } else {
      return buildClustersMR(conf, input, output, measure, t1, t2);
    }
  }

  /**
   * Build a directory of Canopy clusters from the input vectors and other arguments.
   * Run sequential execution
   * 
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measure the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @return the canopy output directory Path
   */
  private static Path buildClustersSeq(Path input, Path output, DistanceMeasure measure, double t1, double t2)
      throws InstantiationException, IllegalAccessException, IOException {
    CanopyClusterer clusterer = new CanopyClusterer(measure, t1, t2);
    Collection<Canopy> canopies = new ArrayList<Canopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        VectorWritable vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        while (reader.next(key, vw)) {
          clusterer.addPointToCanopies(vw.get(), canopies);
          vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
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
        canopy.computeParameters();
        log.info("Writing Canopy:" + canopy.getIdentifier() + " center:" + AbstractCluster.formatVector(canopy.getCenter(), null)
            + " numPoints:" + canopy.getNumPoints() + " radius:" + AbstractCluster.formatVector(canopy.getRadius(), null));
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
   * @param conf 
   * 
   * @param input the Path to the directory containing input vectors
   * @param output the Path for all output directories
   * @param measure the DistanceMeasure 
   * @param t1 the double T1 distance metric
   * @param t2 the double T2 distance metric
   * @return the canopy output directory Path
   */
  private static Path buildClustersMR(Configuration conf, Path input, Path output, DistanceMeasure measure, double t1, double t2)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
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

  public static void clusterData(Configuration conf,
                                 Path points,
                                 Path canopies,
                                 Path output,
                                 DistanceMeasure measure,
                                 double t1,
                                 double t2,
                                 boolean runSequential) throws InstantiationException, IllegalAccessException, IOException,
      InterruptedException, ClassNotFoundException {
    if (runSequential) {
      clusterDataSeq(points, canopies, output, measure, t1, t2);
    } else {
      clusterDataMR(conf, points, canopies, output, measure, t1, t2);
    }
  }

  private static void clusterDataSeq(Path points, Path canopies, Path output, DistanceMeasure measure, double t1, double t2)
      throws InstantiationException, IllegalAccessException, IOException {
    CanopyClusterer clusterer = new CanopyClusterer(measure, t1, t2);

    List<Canopy> clusters = new ArrayList<Canopy>();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(canopies.toUri(), conf);
    FileStatus[] status = fs.listStatus(canopies, new OutputLogFilter());
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      try {
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        Canopy value = reader.getValueClass().asSubclass(Canopy.class).newInstance();
        while (reader.next(key, value)) {
          clusters.add(value);
          value = reader.getValueClass().asSubclass(Canopy.class).newInstance();
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
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        VectorWritable vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        while (reader.next(key, vw)) {
          Canopy closest = clusterer.findClosestCanopy(vw.get(), clusters);
          writer.append(new IntWritable(closest.getId()), new WeightedVectorWritable(1, vw.get()));
          vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }
  }

  private static void clusterDataMR(Configuration conf,
                                    Path points,
                                    Path canopies,
                                    Path output,
                                    DistanceMeasure measure,
                                    double t1,
                                    double t2) throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
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
