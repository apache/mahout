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

package org.apache.mahout.clustering.dirichlet;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.classify.ClusterClassificationDriver;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.DirichletClusteringPolicy;
import org.apache.mahout.clustering.topdown.PathDirectory;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;

public class DirichletDriver extends AbstractJob {
  
  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";
  public static final String MODEL_DISTRIBUTION_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";
  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";
  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";
  public static final String EMIT_MOST_LIKELY_KEY = "org.apache.mahout.clustering.dirichlet.emitMostLikely";
  public static final String THRESHOLD_KEY = "org.apache.mahout.clustering.dirichlet.threshold";
  public static final String MODEL_PROTOTYPE_CLASS_OPTION = "modelPrototype";
  public static final String MODEL_DISTRIBUTION_CLASS_OPTION = "modelDist";
  public static final String ALPHA_OPTION = "alpha";
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new DirichletDriver(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(ALPHA_OPTION, "a0", "The alpha0 value for the DirichletDistribution. Defaults to 1.0", "1.0");
    addOption(MODEL_DISTRIBUTION_CLASS_OPTION, "md",
        "The ModelDistribution class name. Defaults to GaussianClusterDistribution",
        GaussianClusterDistribution.class.getName());
    addOption(MODEL_PROTOTYPE_CLASS_OPTION, "mp",
        "The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector",
        RandomAccessSparseVector.class.getName());
    addOption(DefaultOptionCreator.distanceMeasureOption().withRequired(false).create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.methodOption().create());
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    String modelFactory = getOption(MODEL_DISTRIBUTION_CLASS_OPTION);
    String modelPrototype = getOption(MODEL_PROTOTYPE_CLASS_OPTION);
    String distanceMeasure = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    int numModels = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    double alpha0 = Double.parseDouble(getOption(ALPHA_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    int prototypeSize = readPrototypeSize(input);
    
    DistributionDescription description = new DistributionDescription(modelFactory, modelPrototype, distanceMeasure,
        prototypeSize);
    
    run(getConf(), input, output, description, numModels, maxIterations, alpha0, runClustering, emitMostLikely,
        threshold, runSequential);
    return 0;
  }
  
  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   * 
   * @param conf
   *          the Configuration to use
   * @param input
   *          the directory Path for input points
   * @param output
   *          the directory Path for output points
   * @param description
   *          model distribution parameters
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param runClustering
   *          true if clustering of points to be done after iterations
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential
   *          execute sequentially if true
   */
  public static void run(Configuration conf, Path input, Path output, DistributionDescription description,
      int numModels, int maxIterations, double alpha0, boolean runClustering, boolean emitMostLikely, double threshold,
      boolean runSequential) throws IOException, ClassNotFoundException, InterruptedException {
    Path clustersOut = buildClusters(conf, input, output, description, numModels, maxIterations, alpha0, runSequential);
    if (runClustering) {
      clusterData(conf, input, clustersOut, output, alpha0, numModels, emitMostLikely, threshold, runSequential);
    }
  }
  
  /**
   * Read the first input vector to determine the prototype size for the modelPrototype
   */
  public static int readPrototypeSize(Path input) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, PathFilters.logsCRCFilter());
    int protoSize = 0;
    if (status.length > 0) {
      FileStatus s = status[0];
      for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(s.getPath(), true, conf)) {
        protoSize = value.get().size();
      }
    }
    return protoSize;
  }
  
  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * 
   * @param conf
   *          the hadoop configuration
   * @param input
   *          the directory Path for input points
   * @param output
   *          the directory Path for output points
   * @param description
   *          model distribution parameters
   * @param numClusters
   *          the number of models to iterate over
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param runSequential
   *          execute sequentially if true
   * 
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf, Path input, Path output, DistributionDescription description,
      int numClusters, int maxIterations, double alpha0, boolean runSequential) throws IOException,
      ClassNotFoundException, InterruptedException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    ModelDistribution<VectorWritable> modelDist = description.createModelDistribution(conf);
    
    List<Cluster> models = Lists.newArrayList();
    for (Model<VectorWritable> cluster : modelDist.sampleFromPrior(numClusters)) {
      models.add((Cluster) cluster);
    }
    
    ClusterClassifier prior = new ClusterClassifier(models, new DirichletClusteringPolicy(numClusters, alpha0));
    prior.writeToSeqFiles(clustersIn);
    
    if (runSequential) {
      ClusterIterator.iterateSeq(conf, input, clustersIn, output, maxIterations);
    } else {
      ClusterIterator.iterateMR(conf, input, clustersIn, output, maxIterations);
    }
    return output;
    
  }
  
  /**
   * Run the job using supplied arguments
   * 
   * @param conf
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param output
   *          the directory pathname for output points
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential
   *          execute sequentially if true
   */
  public static void clusterData(Configuration conf, Path input, Path stateIn, Path output, double alpha0,
      int numModels, boolean emitMostLikely, double threshold, boolean runSequential) throws IOException,
      InterruptedException, ClassNotFoundException {
    ClusterClassifier.writePolicy(new DirichletClusteringPolicy(numModels, alpha0), stateIn);
    ClusterClassificationDriver.run(conf, input, output, new Path(output, PathDirectory.CLUSTERED_POINTS_DIRECTORY), threshold,
        emitMostLikely, runSequential);
  }
  
}
