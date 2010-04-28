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

package org.apache.mahout.clustering.cdbw;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.WeightedPointWritable;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CDbwDriver {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  private static final Logger log = LoggerFactory.getLogger(CDbwDriver.class);

  private CDbwDriver() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();

    Option modelOpt = obuilder.withLongName("modelClass").withRequired(true).withShortName("d").withArgument(
        abuilder.withName("modelClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The ModelDistribution class name. " + "Defaults to org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution")
        .create();

    Option numRedOpt = obuilder.withLongName("maxRed").withRequired(true).withShortName("r").withArgument(
        abuilder.withName("maxRed").withMinimum(1).withMaximum(1).create()).withDescription("The number of reduce tasks.").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(modelOpt).withOption(
        maxIterOpt).withOption(helpOpt).withOption(numRedOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      String input = cmdLine.getValue(inputOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      String modelFactory = "org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution";
      if (cmdLine.hasOption(modelOpt)) {
        modelFactory = cmdLine.getValue(modelOpt).toString();
      }
      int numReducers = Integer.parseInt(cmdLine.getValue(numRedOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      runJob(input, null, output, modelFactory, maxIterations, numReducers);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param clustersIn
   *          the directory pathname for input [n/a :: Cluster]
   * @param clusteredPointsIn 
              the directory pathname for input clustered points [clusterId :: VectorWritable]
   * @param output
   *          the directory pathname for output reference points [clusterId :: VectorWritable]
   * @param distanceMeasureClass
   *          the String ModelDistribution class name to use
   * @param numIterations
   *          the number of iterations
   * @param numReducers
   *          the number of Reducers desired
   */
  public static void runJob(String clustersIn, String clusteredPointsIn, String output, String distanceMeasureClass,
      int numIterations, int numReducers) throws ClassNotFoundException, InstantiationException, IllegalAccessException,
      IOException, SecurityException, NoSuchMethodException, InvocationTargetException {

    String stateIn = output + "/representativePoints-0";
    writeInitialState(stateIn, clustersIn);

    for (int iteration = 0; iteration < numIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String stateOut = output + "/representativePoints-" + (iteration + 1);
      runIteration(clusteredPointsIn, stateIn, stateOut, distanceMeasureClass, numReducers);
      // now point the input to the old output directory
      stateIn = stateOut;
    }

    Configurable client = new JobClient();
    JobConf conf = new JobConf(CDbwDriver.class);
    conf.set(STATE_IN_KEY, stateIn);
    conf.set(DISTANCE_MEASURE_KEY, distanceMeasureClass);
    CDbwEvaluator evaluator = new CDbwEvaluator(conf, clustersIn);
    System.out.println("CDbw = " + evaluator.CDbw());
  }

  private static void writeInitialState(String output, String clustersIn) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException, SecurityException, NoSuchMethodException, InvocationTargetException {

    JobConf job = new JobConf(KMeansDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), job);
    File f = new File(clustersIn);
    for (File part : f.listFiles()) {
      if (!part.getName().startsWith(".")) {
        Path inPart = new Path(clustersIn + "/" + part.getName());
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, inPart, job);
        Writable key = (Writable) reader.getKeyClass().newInstance();
        Writable value = (Writable) reader.getValueClass().newInstance();
        Path path = new Path(output + "/" + part.getName());
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, IntWritable.class, VectorWritable.class);
        while (reader.next(key, value)) {
          Cluster cluster = (Cluster) value;
          if (!(cluster instanceof DirichletCluster) || ((DirichletCluster) cluster).getTotalCount() > 0) {
            System.out.println("C-" + cluster.getId() + ": " + ClusterBase.formatVector(cluster.getCenter(), null));
            writer.append(new IntWritable(cluster.getId()), new VectorWritable(cluster.getCenter()));
          }
        }
        writer.close();
      }
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param stateOut
   *          the directory pathname for output state
   * @param distanceMeasureClass
   *          the class name of the DistanceMeasure class
   * @param numReducers
   *          the number of Reducers desired
   */
  public static void runIteration(String input, String stateIn, String stateOut, String distanceMeasureClass, int numReducers) {
    Configurable client = new JobClient();
    JobConf conf = new JobConf(CDbwDriver.class);

    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(WeightedPointWritable.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(stateOut);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(CDbwMapper.class);
    conf.setReducerClass(CDbwReducer.class);
    conf.setNumReduceTasks(numReducers);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(STATE_IN_KEY, stateIn);
    conf.set(DISTANCE_MEASURE_KEY, distanceMeasureClass);

    client.setConf(conf);
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }
}
