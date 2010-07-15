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
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
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
   * Run the job
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
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void runJob(Path input, Path output, String measureClassName, double t1, double t2, boolean runClustering)
      throws IOException, InterruptedException, ClassNotFoundException {
    new CanopyDriver().job(input, output, measureClassName, t1, t2, runClustering);
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

    Map<String, String> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (argMap.containsKey(DefaultOptionCreator.OVERWRITE_OPTION_KEY)) {
      HadoopUtil.overwriteOutput(output);
    }
    String measureClass = argMap.get(DefaultOptionCreator.DISTANCE_MEASURE_OPTION_KEY);
    double t1 = Double.parseDouble(argMap.get(DefaultOptionCreator.T1_OPTION_KEY));
    double t2 = Double.parseDouble(argMap.get(DefaultOptionCreator.T2_OPTION_KEY));
    boolean runClustering = argMap.containsKey(DefaultOptionCreator.CLUSTERING_OPTION_KEY);

    job(input, output, measureClass, t1, t2, runClustering);
    return 0;
  }

  private void job(Path input, Path output, String measureClassName, double t1, double t2, boolean runClustering)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("Input: {} Out: {} " + "Measure: {} t1: {} t2: {}", new Object[] { input, output, measureClassName, t1, t2 });
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

    if (runClustering) {
      runClustering(input, canopyOutputDir, output, measureClassName, t1, t2);
    }
  }

  /**
   * Run the job
   * 
   * @param points
   *          the input points directory pathname String
   * @param canopies
   *          the input canopies directory pathname String
   * @param output
   *          the output directory pathname String
   * @param measureClassName
   *          the DistanceMeasure class name
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private void runClustering(Path points, Path canopies, Path output, String measureClassName, double t1, double t2)
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
