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

package org.apache.mahout.clustering.spectral.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.spectral.common.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.common.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.common.UnitVectorizerJob;
import org.apache.mahout.clustering.spectral.common.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Implementation of the EigenCuts spectral clustering algorithm.
 */
public class SpectralKMeansDriver extends AbstractJob {

  public static final double OVERSHOOT_MULTIPLIER = 2.0;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SpectralKMeansDriver(), args);
  }

  @Override
  public int run(String[] arg0) throws IOException, ClassNotFoundException, InterruptedException {
    // set up command line options
    Configuration conf = getConf();
    addInputOption();
    addOutputOption();
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("clusters", "k", "Number of clusters and top eigenvectors", true);
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    Map<String, List<String>> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(conf, output);
    }
    int numDims = Integer.parseInt(getOption("dimensions"));
    int clusters = Integer.parseInt(getOption("clusters"));
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));

    run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations);

    return 0;
  }

  /**
   * Run the Spectral KMeans clustering on the supplied arguments
   * 
   * @param conf the Configuration to be used
   * @param input the Path to the input tuples directory
   * @param output the Path to the output directory
   * @param numDims the int number of dimensions of the affinity matrix
   * @param clusters the int number of eigenvectors and thus clusters to produce
   * @param measure the DistanceMeasure for the k-Means calculations
   * @param convergenceDelta the double convergence delta for the k-Means calculations
   * @param maxIterations the int maximum number of iterations for the k-Means calculations
   */
  public static void run(Configuration conf,
                         Path input,
                         Path output,
                         int numDims,
                         int clusters,
                         DistanceMeasure measure,
                         double convergenceDelta,
                         int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    // create a few new Paths for temp files and transformations
    Path outputCalc = new Path(output, "calculations");
    Path outputTmp = new Path(output, "temporary");

    // Take in the raw CSV text file and split it ourselves,
    // creating our own SequenceFiles for the matrices to read later 
    // (similar to the style of syntheticcontrol.canopy.InputMapper)
    Path affSeqFiles = new Path(outputCalc, "seqfile-" + (System.nanoTime() & 0xFF));
    AffinityMatrixInputJob.runJob(input, affSeqFiles, numDims, numDims);

    // Next step: construct the affinity matrix using the newly-created
    // sequence files
    DistributedRowMatrix A = new DistributedRowMatrix(affSeqFiles,
                                                      new Path(outputTmp, "afftmp-" + (System.nanoTime() & 0xFF)),
                                                      numDims,
                                                      numDims);
    Configuration depConf = new Configuration(conf);
    A.setConf(depConf);

    // Next step: construct the diagonal matrix D (represented as a vector)
    // and calculate the normalized Laplacian of the form:
    // L = D^(-0.5)AD^(-0.5)
    Vector D = MatrixDiagonalizeJob.runJob(affSeqFiles, numDims);
    DistributedRowMatrix L =
        VectorMatrixMultiplicationJob.runJob(affSeqFiles, D,
            new Path(outputCalc, "laplacian-" + (System.nanoTime() & 0xFF)), new Path(outputCalc, "laplacian-tmp-" + (System.nanoTime() & 0xFF)));
    L.setConf(depConf);

    // Next step: perform eigen-decomposition using LanczosSolver
    // since some of the eigen-output is spurious and will be eliminated
    // upon verification, we have to aim to overshoot and then discard
    // unnecessary vectors later
    int overshoot = (int) ((double) clusters * OVERSHOOT_MULTIPLIER);
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    LanczosState state = new LanczosState(L, clusters, DistributedLanczosSolver.getInitialVector(L));
    Path lanczosSeqFiles = new Path(outputCalc, "eigenvectors-" + (System.nanoTime() & 0xFF));
    solver.runJob(conf,
                  state,
                  overshoot,
                  true,
                  lanczosSeqFiles.toString());

    // perform a verification
    EigenVerificationJob verifier = new EigenVerificationJob();
    Path verifiedEigensPath = new Path(outputCalc, "eigenverifier");
    verifier.runJob(conf, lanczosSeqFiles, L.getRowPath(), verifiedEigensPath, true, 1.0, clusters);
    Path cleanedEigens = verifier.getCleanedEigensPath();
    DistributedRowMatrix W = new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens, "tmp"), clusters, numDims);
    W.setConf(depConf);
    DistributedRowMatrix Wtrans = W.transpose();
    //    DistributedRowMatrix Wt = W.transpose();

    // next step: normalize the rows of Wt to unit length
    Path unitVectors = new Path(outputCalc, "unitvectors-" + (System.nanoTime() & 0xFF));
    UnitVectorizerJob.runJob(Wtrans.getRowPath(), unitVectors);
    DistributedRowMatrix Wt = new DistributedRowMatrix(unitVectors, new Path(unitVectors, "tmp"), clusters, numDims);
    Wt.setConf(depConf);

    // Finally, perform k-means clustering on the rows of L (or W)
    // generate random initial clusters
    Path initialclusters = RandomSeedGenerator.buildRandom(conf,
                                                           Wt.getRowPath(),
                                                           new Path(output, Cluster.INITIAL_CLUSTERS_DIR),
                                                           clusters,
                                                           measure);
    
    // The output format is the same as the K-means output format.
    // TODO: Perhaps a conversion of the output format from points and clusters
    // in eigenspace to the original dataset. Currently, the user has to perform
    // the association step after this job finishes on their own.
    KMeansDriver.run(conf,
                     Wt.getRowPath(),
                     initialclusters,
                     output,
                     measure,
                     convergenceDelta,
                     maxIterations,
                     true,
                     0.0, 
                     false);
  }
}
