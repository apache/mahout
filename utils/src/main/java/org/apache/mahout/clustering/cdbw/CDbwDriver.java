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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class CDbwDriver extends AbstractJob {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  //public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  private static final Logger log = LoggerFactory.getLogger(CDbwDriver.class);

  private CDbwDriver() {
  }

  public static void main(String[] args) throws Exception {
    new CDbwDriver().run(args);
  }

  @Override
  public int run(String[] args)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.numReducersOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    String distanceMeasureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    int numReducers = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    job(input, null, output, distanceMeasureClass, maxIterations, numReducers);
    return 0;
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
   * @throws InterruptedException 
   */
  public static void runJob(Path clustersIn,
                            Path clusteredPointsIn,
                            Path output,
                            String distanceMeasureClass,
                            int numIterations,
                            int numReducers)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException,
        IOException, InterruptedException {
    job(clustersIn, clusteredPointsIn, output, distanceMeasureClass, numIterations, numReducers);
  }

  private static void job(Path clustersIn,
                          Path clusteredPointsIn,
                          Path output,
                          String distanceMeasureClass,
                          int numIterations,
                          int numReducers)
      throws InstantiationException, IllegalAccessException, IOException, InterruptedException, ClassNotFoundException {
    Path stateIn = new Path(output, "representativePoints-0");
    writeInitialState(stateIn, clustersIn);

    for (int iteration = 0; iteration < numIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path stateOut = new Path(output, "representativePoints-" + (iteration + 1));
      runIteration(clusteredPointsIn, stateIn, stateOut, distanceMeasureClass, numReducers);
      // now point the input to the old output directory
      stateIn = stateOut;
    }

    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(DISTANCE_MEASURE_KEY, distanceMeasureClass);
    CDbwEvaluator evaluator = new CDbwEvaluator(conf, clustersIn);
    // now print out the Results
    System.out.println("CDbw = " + evaluator.CDbw());
    System.out.println("Intra-cluster density = " + evaluator.intraClusterDensity());
    System.out.println("Inter-cluster density = " + evaluator.interClusterDensity());
    System.out.println("Separation = " + evaluator.separation());
  }

  private static void writeInitialState(Path output, Path clustersIn) throws InstantiationException, IllegalAccessException,
      IOException, SecurityException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    for (FileStatus part : fs.listStatus(clustersIn)) {
      if (!part.getPath().getName().startsWith(".")) {
        Path inPart = part.getPath();
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, inPart, conf);
        Writable key = (Writable) reader.getKeyClass().newInstance();
        Writable value = (Writable) reader.getValueClass().newInstance();
        Path path = new Path(output, inPart.getName());
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
        while (reader.next(key, value)) {
          Cluster cluster = (Cluster) value;
          if (!(cluster instanceof DirichletCluster<?>) || ((DirichletCluster<?>) cluster).getTotalCount() > 0) {
            //System.out.println("C-" + cluster.getId() + ": " + ClusterBase.formatVector(cluster.getCenter(), null));
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
  private static void runIteration(Path input,
                                   Path stateIn,
                                   Path stateOut,
                                   String distanceMeasureClass,
                                   int numReducers)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(DISTANCE_MEASURE_KEY, distanceMeasureClass);
    Job job = new Job(conf);
    job.setJarByClass(CDbwDriver.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, stateOut);

    job.setMapperClass(CDbwMapper.class);
    job.setReducerClass(CDbwReducer.class);
    job.setNumReduceTasks(numReducers);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.waitForCompletion(true);
  }
}
