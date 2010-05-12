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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class CanopyDriver {

  private static final Logger log = LoggerFactory.getLogger(CanopyDriver.class);

  public static final String DEFAULT_CLUSTERED_POINTS_DIRECTORY = "clusteredPoints";

  private CanopyDriver() {
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription("The Path to put the output in")
        .withShortName("o").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
        "If set, overwrite the output directory").withShortName("w").create();

    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Distance Measure to use.  Default is SquaredEuclidean").withShortName("m").create();

    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Vector implementation class name.  Default is RandomAccessSparseVector.class").withShortName("v").create();
    Option t1Opt = obuilder.withLongName("t1").withRequired(true).withArgument(
        abuilder.withName("t1").withMinimum(1).withMaximum(1).create()).withDescription("t1").withShortName("t1").create();
    Option t2Opt = obuilder.withLongName("t2").withRequired(true).withArgument(
        abuilder.withName("t2").withMinimum(1).withMaximum(1).create()).withDescription("t2").withShortName("t2").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(overwriteOutput).withOption(
        measureClassOpt).withOption(vectorClassOpt).withOption(t1Opt).withOption(t2Opt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      String measureClass = SquaredEuclideanDistanceMeasure.class.getName();
      if (cmdLine.hasOption(measureClassOpt)) {
        measureClass = cmdLine.getValue(measureClassOpt).toString();
      }

      // Class<? extends Vector> vectorClass = cmdLine.hasOption(vectorClassOpt) == false ?
      // RandomAccessSparseVector.class
      // : (Class<? extends Vector>) Class.forName(cmdLine.getValue(vectorClassOpt).toString());
      double t1 = Double.parseDouble(cmdLine.getValue(t1Opt).toString());
      double t2 = Double.parseDouble(cmdLine.getValue(t2Opt).toString());

      runJob(input, output, measureClass, t1, t2, false);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);

    }
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
   */
  public static void runJob(Path input, Path output, String measureClassName, double t1, double t2, boolean runClustering)
      throws IOException {
    log.info("Input: {} Out: {} " + "Measure: {} t1: {} t2: {}", new Object[] { input, output, measureClassName, t1, t2 });
    Configurable client = new JobClient();
    JobConf conf = new JobConf(CanopyDriver.class);
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(t2));

    conf.setInputFormat(SequenceFileInputFormat.class);

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Canopy.class);

    FileInputFormat.setInputPaths(conf, input);

    Path canopyOutputDir = new Path(output, Cluster.CLUSTERS_DIR + '0');
    FileOutputFormat.setOutputPath(conf, canopyOutputDir);

    conf.setMapperClass(CanopyMapper.class);
    conf.setReducerClass(CanopyReducer.class);
    conf.setNumReduceTasks(1);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    client.setConf(conf);

    JobClient.runJob(conf);

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
   */
  public static void runClustering(Path points, Path canopies, Path output, String measureClassName, double t1, double t2)
      throws IOException {
    Configurable client = new JobClient();
    JobConf conf = new JobConf(CanopyDriver.class);

    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(t1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(t2));
    conf.set(CanopyConfigKeys.CANOPY_PATH_KEY, canopies.toString());

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(WeightedVectorWritable.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(conf, points);
    Path outPath = new Path(output, DEFAULT_CLUSTERED_POINTS_DIRECTORY);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(ClusterMapper.class);
    conf.setReducerClass(IdentityReducer.class);
    conf.setNumReduceTasks(0);

    client.setConf(conf);
    HadoopUtil.overwriteOutput(outPath);
    JobClient.runJob(conf);
  }

}
