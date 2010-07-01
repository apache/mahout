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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
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
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MeanShiftCanopyDriver {

  private static final Logger log = LoggerFactory.getLogger(MeanShiftCanopyDriver.class);

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.meanshift.stateInKey";

  private static final String CONTROL_CONVERGED = "control/converged";

  private MeanShiftCanopyDriver() {
  }

  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option maxIterOpt = DefaultOptionCreator.maxIterationsOption().create();
    Option overwriteOutput = DefaultOptionCreator.overwriteOption().create();
    Option inputIsCanopiesOpt = DefaultOptionCreator.inputIsCanopiesOption().create();
    Option measureClassOpt = DefaultOptionCreator.distanceMeasureOption().create();
    Option threshold1Opt = DefaultOptionCreator.t1Option().create();
    Option threshold2Opt = DefaultOptionCreator.t2Option().create();
    Option clusteringOpt = DefaultOptionCreator.clusteringOption().create();

    Group group = new GroupBuilder().withName("Options").withOption(inputOpt).withOption(outputOpt)
        .withOption(overwriteOutput).withOption(measureClassOpt).withOption(helpOpt)
        .withOption(convergenceDeltaOpt).withOption(threshold1Opt).withOption(threshold2Opt)
        .withOption(clusteringOpt).withOption(maxIterOpt).withOption(inputIsCanopiesOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());
      String measureClass = cmdLine.getValue(measureClassOpt).toString();
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      double t1 = Double.parseDouble(cmdLine.getValue(threshold1Opt).toString());
      double t2 = Double.parseDouble(cmdLine.getValue(threshold2Opt).toString());
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      runJob(input, output, measureClass, t1, t2, convergenceDelta, maxIterations,
             cmdLine.hasOption(inputIsCanopiesOpt), cmdLine.hasOption(clusteringOpt));
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
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
  static void runIteration(Path input, Path output, Path control, String measureClassName, double t1, double t2,
      double convergenceDelta) throws IOException, InterruptedException, ClassNotFoundException {

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

    job.waitForCompletion(true);
  }

  static void createCanopyFromVectors(Path input, Path output)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    Job job = new Job(conf);
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
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output clustered points
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IOException 
   */
  static void runClustering(Path input, Path clustersIn, Path output)
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

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.waitForCompletion(true);
  }

  /**
   * Run the job where the input format can be either Vectors or Canopies
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
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void runJob(Path input,
                            Path output,
                            String measureClassName,
                            double t1,
                            double t2,
                            double convergenceDelta,
                            int maxIterations,
                            boolean inputIsCanopies,
                            boolean runClustering)
    throws IOException, InterruptedException, ClassNotFoundException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    if (inputIsCanopies) {
      clustersIn = input;
    } else {
      createCanopyFromVectors(input, clustersIn);
    }

    // iterate until the clusters converge
    boolean converged = false;
    int iteration = 1;
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      Path controlOut = new Path(output, CONTROL_CONVERGED);
      runIteration(clustersIn, clustersOut, controlOut, measureClassName, t1, t2, convergenceDelta);
      converged = FileSystem.get(new Configuration()).exists(controlOut);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }

    if (runClustering) {
      // now cluster the points
      runClustering(inputIsCanopies ? input : new Path(output, Cluster.INITIAL_CLUSTERS_DIR), clustersIn, new Path(output,
          Cluster.CLUSTERED_POINTS_DIR));
    }
  }
}
