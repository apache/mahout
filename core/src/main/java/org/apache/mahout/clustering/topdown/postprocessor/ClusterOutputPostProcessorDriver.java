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

package org.apache.mahout.clustering.topdown.postprocessor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

/**
 * Post processes the output of clustering algorithms and groups them into respective clusters. Ideal to be
 * used for top down clustering. It can also be used if the clustering output needs to be grouped into their
 * respective clusters.
 */
public final class ClusterOutputPostProcessorDriver extends AbstractJob {

  /**
   * CLI to run clustering post processor. The input to post processor is the ouput path specified to the
   * clustering.
   */
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
            DefaultOptionCreator.SEQUENTIAL_METHOD);
    run(input, output, runSequential);
    return 0;

  }

  /**
   * Constructor to be used by the ToolRunner.
   */
  private ClusterOutputPostProcessorDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new ClusterOutputPostProcessorDriver(), args);
  }

  /**
   * Post processes the output of clustering algorithms and groups them into respective clusters. Each
   * cluster's vectors are written into a directory named after its clusterId.
   *
   * @param input         The output path provided to the clustering algorithm, whose would be post processed. Hint: The
   *                      path of the directory containing clusters-*-final and clusteredPoints.
   * @param output        The post processed data would be stored at this path.
   * @param runSequential If set to true, post processes it sequentially, else, uses. MapReduce. Hint: If the clustering
   *                      was done sequentially, make it sequential, else vice versa.
   */
  public static void run(Path input, Path output, boolean runSequential) throws IOException,
          InterruptedException,
          ClassNotFoundException {
    if (runSequential) {
      postProcessSeq(input, output);
    } else {
      Configuration conf = new Configuration();
      postProcessMR(conf, input, output);
      movePartFilesToRespectiveDirectories(conf, output);
    }

  }

  /**
   * Process Sequentially. Reads the vectors one by one, and puts them into respective directory, named after
   * their clusterId.
   *
   * @param input  The output path provided to the clustering algorithm, whose would be post processed. Hint : The
   *               path of the directory containing clusters-*-final and clusteredPoints.
   * @param output The post processed data would be stored at this path.
   */
  private static void postProcessSeq(Path input, Path output) throws IOException {
    ClusterOutputPostProcessor clusterOutputPostProcessor = new ClusterOutputPostProcessor(input, output,
            new Configuration());
    clusterOutputPostProcessor.process();
  }

  /**
   * Process as a map reduce job. The numberOfReduceTasks is set to the number of clusters present in the
   * output. So that each cluster's vector is written in its own part file.
   *
   * @param conf   The hadoop configuration.
   * @param input  The output path provided to the clustering algorithm, whose would be post processed. Hint : The
   *               path of the directory containing clusters-*-final and clusteredPoints.
   * @param output The post processed data would be stored at this path.
   */
  private static void postProcessMR(Configuration conf, Path input, Path output) throws IOException,
          InterruptedException,
          ClassNotFoundException {
    System.out.println("WARNING: If you are running in Hadoop local mode, please use the --sequential option, "
        + "as the MapReduce option will not work properly");
    int numberOfClusters = ClusterCountReader.getNumberOfClusters(input, conf);
    conf.set("clusterOutputPath", input.toString());
    Job job = new Job(conf, "ClusterOutputPostProcessor Driver running over input: " + input);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(ClusterOutputPostProcessorMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setReducerClass(ClusterOutputPostProcessorReducer.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(numberOfClusters);
    job.setJarByClass(ClusterOutputPostProcessorDriver.class);

    FileInputFormat.addInputPath(job, new Path(input, new Path("clusteredPoints")));
    FileOutputFormat.setOutputPath(job, output);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("ClusterOutputPostProcessor Job failed processing " + input);
    }
  }

  /**
   * The mapreduce version of the post processor writes different clusters into different part files. This
   * method reads the part files and moves them into directories named after their clusterIds.
   *
   * @param conf   The hadoop configuration.
   * @param output The post processed data would be stored at this path.
   */
  private static void movePartFilesToRespectiveDirectories(Configuration conf, Path output) throws IOException {
    FileSystem fileSystem = output.getFileSystem(conf);
    for (FileStatus fileStatus : fileSystem.listStatus(output, PathFilters.partFilter())) {
      SequenceFileIterator<Writable, Writable> it =
              new SequenceFileIterator<Writable, Writable>(fileStatus.getPath(), true, conf);
      if (it.hasNext()) {
        renameFile(it.next().getFirst(), fileStatus, conf);
      }
      it.close();
    }
  }

  /**
   * Using @FileSystem rename method to move the file.
   */
  private static void renameFile(Writable key, FileStatus fileStatus, Configuration conf) throws IOException {
    Path path = fileStatus.getPath();
    FileSystem fileSystem = path.getFileSystem(conf);
    Path subDir = new Path(key.toString());
    Path renameTo = new Path(path.getParent(), subDir);
    fileSystem.mkdirs(renameTo);
    fileSystem.rename(path, renameTo);
  }

}
