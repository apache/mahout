/*
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

package org.apache.mahout.clustering.syntheticcontrol.dirichlet;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.DirichletMapper;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.clustering.syntheticcontrol.canopy.InputDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends DirichletDriver {

  private Job() {
    super();
  }

  private static final Logger log = LoggerFactory.getLogger(Job.class);

  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      new Job().run(args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      HadoopUtil.overwriteOutput(output);
      new Job().job(new Path("testdata"),
                    output,
                    "org.apache.mahout.clustering.syntheticcontrol.dirichlet.NormalScModelDistribution",
                    "org.apache.mahout.math.RandomAccessSparseVector",
                    10,
                    5,
                    1.0,
                    1,
                    false,
                    0.001);
    }
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
   */
  public int run(String[] args) throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException,
      NoSuchMethodException, InvocationTargetException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(new DefaultOptionBuilder().withLongName(ALPHA_OPTION).withRequired(false).withShortName("m")
        .withArgument(new ArgumentBuilder().withName(ALPHA_OPTION).withDefault("1.0").withMinimum(1).withMaximum(1).create())
        .withDescription("The alpha0 value for the DirichletDistribution. Defaults to 1.0").create());
    addOption(new DefaultOptionBuilder().withLongName(MODEL_DISTRIBUTION_CLASS_OPTION).withRequired(false).withShortName("md")
        .withArgument(new ArgumentBuilder().withName(MODEL_DISTRIBUTION_CLASS_OPTION).withDefault(NormalModelDistribution.class
            .getName()).withMinimum(1).withMaximum(1).create()).withDescription("The ModelDistribution class name. "
            + "Defaults to NormalModelDistribution").create());
    addOption(new DefaultOptionBuilder().withLongName(MODEL_PROTOTYPE_CLASS_OPTION).withRequired(false).withShortName("mp")
        .withArgument(new ArgumentBuilder().withName("prototypeClass").withDefault(RandomAccessSparseVector.class.getName())
            .withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector").create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.numReducersOption().create());

    Map<String, String> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    String modelFactory = getOption(MODEL_DISTRIBUTION_CLASS_OPTION);
    String modelPrototype = getOption(MODEL_PROTOTYPE_CLASS_OPTION);
    int numModels = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
    int numReducers = Integer.parseInt(getOption(DefaultOptionCreator.MAX_REDUCERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    double alpha0 = Double.parseDouble(getOption(ALPHA_OPTION));

    job(input, output, modelFactory, modelPrototype, numModels, maxIterations, alpha0, numReducers, emitMostLikely, threshold);
    return 0;
  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it exists beforehand
   * 
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   * @param modelFactory
   *          the ModelDistribution class name
   * @param numModels
   *          the number of Models
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha0 value for the DirichletDistribution
   * @param numReducers
   *          the desired number of reducers
   * @param emitMostLikely 
   * @param threshold 
   * @throws InterruptedException 
   * @throws SecurityException 
   */
  private void job(Path input,
                   Path output,
                   String modelFactory,
                   String modelPrototype,
                   int numModels,
                   int maxIterations,
                   double alpha0,
                   int numReducers,
                   boolean emitMostLikely,
                   double threshold) throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException,
      NoSuchMethodException, InvocationTargetException, SecurityException, InterruptedException {
    Path directoryContainingConvertedInput = new Path(output, Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT);
    InputDriver.runJob(input, directoryContainingConvertedInput, modelPrototype);
    DirichletDriver.runJob(directoryContainingConvertedInput,
                           output,
                           modelFactory,
                           modelPrototype,
                           numModels,
                           maxIterations,
                           alpha0,
                           numReducers,
                           true,
                           emitMostLikely,
                           threshold);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(new Path(output, "clusters-" + maxIterations), new Path(output,
                                                                                                            "clusteredPoints"));
    clusterDumper.printClusters(null);
  }

  /**
   * Prints out all of the clusters for each iteration
   * 
   * @param output
   *          the String output directory
   * @param modelDistribution
   *          the String class name of the ModelDistribution
   * @param vectorClassName
   *          the String class name of the Vector to use
   * @param prototypeSize
   *          the size of the Vector prototype for the Dirichlet Models
   * @param numIterations
   *          the int number of Iterations
   * @param numModels
   *          the int number of models
   * @param alpha0
   *          the double alpha_0 value
   * @throws InvocationTargetException
   * @throws NoSuchMethodException
   * @throws SecurityException
   */
  public static void printResults(String output,
                                  String modelDistribution,
                                  String vectorClassName,
                                  int prototypeSize,
                                  int numIterations,
                                  int numModels,
                                  double alpha0) throws NoSuchMethodException, InvocationTargetException {
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_FACTORY_KEY, modelDistribution);
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(numModels));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(alpha0));
    for (int i = 0; i < numIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, output + "/clusters-" + i);
      conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, vectorClassName);
      conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, Integer.toString(prototypeSize));
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printClusters(clusters, 0);

  }

  /**
   * Actually prints out the clusters
   * 
   * @param clusters
   *          a List of Lists of DirichletClusters
   * @param significant
   *          the minimum number of samples to enable printing a model
   */
  private static void printClusters(List<List<DirichletCluster<VectorWritable>>> clusters, int significant) {
    int row = 0;
    StringBuilder result = new StringBuilder();
    for (List<DirichletCluster<VectorWritable>> r : clusters) {
      result.append("sample=").append(row++).append("]= ");
      for (int k = 0; k < r.size(); k++) {
        Model<VectorWritable> model = r.get(k).getModel();
        if (model.count() > significant) {
          int total = (int) r.get(k).getTotalCount();
          result.append('m').append(k).append('(').append(total).append(')').append(model).append(", ");
        }
      }
      result.append('\n');
    }
    result.append('\n');
    log.info(result.toString());
  }
}
