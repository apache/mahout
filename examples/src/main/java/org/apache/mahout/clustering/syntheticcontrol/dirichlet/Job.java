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

import java.util.List;
import java.util.Map;

import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(Job.class);

  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";

  private Job() {
  }

  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      ToolRunner.run(new Configuration(), new Job(), args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      HadoopUtil.delete(new Configuration(), output);
      DistributionDescription description =
          new DistributionDescription(GaussianClusterDistribution.class.getName(),
                                      RandomAccessSparseVector.class.getName(),
                                      null,
                                      60);
      run(new Path("testdata"), output, description, 10, 5, 1.0, true, 0);
    }
  }

  @Override
  public int run(String[] args) throws Exception{
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(new DefaultOptionBuilder().withLongName(DirichletDriver.ALPHA_OPTION).withRequired(false)
        .withShortName("m").withArgument(new ArgumentBuilder().withName(DirichletDriver.ALPHA_OPTION).withDefault("1.0")
            .withMinimum(1).withMaximum(1).create())
        .withDescription("The alpha0 value for the DirichletDistribution. Defaults to 1.0").create());
    addOption(new DefaultOptionBuilder().withLongName(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION)
        .withRequired(false).withShortName("md").withArgument(new ArgumentBuilder()
            .withName(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION)
            .withDefault(GaussianClusterDistribution.class.getName()).withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution class name. Defaults to GaussianClusterDistribution").create());
    addOption(new DefaultOptionBuilder().withLongName(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION).withRequired(false)
        .withShortName("mp").withArgument(new ArgumentBuilder().withName("prototypeClass")
            .withDefault(RandomAccessSparseVector.class.getName()).withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector")
        .create());
    addOption(DefaultOptionCreator.distanceMeasureOption().withRequired(false).create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());

    Map<String, List<String>> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    String modelFactory = getOption(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION);
    String modelPrototype = getOption(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION);
    String distanceMeasure = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    int numModels = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    double alpha0 = Double.parseDouble(getOption(DirichletDriver.ALPHA_OPTION));
    DistributionDescription description =
        new DistributionDescription(modelFactory, modelPrototype, distanceMeasure, 60);

    run(input, output, description, numModels, maxIterations, alpha0, emitMostLikely, threshold);
    return 0;
  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it exists beforehand
   * 
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   * @param description the model distribution description
   * @param numModels
   *          the number of Models
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha0 value for the DirichletDistribution
   */
  public static void run(Path input,
                         Path output,
                         DistributionDescription description,
                         int numModels,
                         int maxIterations,
                         double alpha0,
                         boolean emitMostLikely,
                         double threshold)
          throws Exception{
    Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
    DirichletDriver.run(new Configuration(), directoryContainingConvertedInput, output, description, numModels, maxIterations, alpha0, true,
    emitMostLikely, threshold, false);
    // run ClusterDumper
    ClusterDumper clusterDumper =
        new ClusterDumper(new Path(output, "clusters-" + maxIterations), new Path(output, "clusteredPoints"));
    clusterDumper.printClusters(null);
  }

  /**
   * Actually prints out the clusters
   * 
   * @param clusters
   *          a List of Lists of DirichletClusters
   * @param significant
   *          the minimum number of samples to enable printing a model
   */
  /*
  private static void printClusters(Iterable<List<DirichletCluster>> clusters, int significant) {
    int row = 0;
    StringBuilder result = new StringBuilder(100);
    for (List<DirichletCluster> r : clusters) {
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
   */
}
