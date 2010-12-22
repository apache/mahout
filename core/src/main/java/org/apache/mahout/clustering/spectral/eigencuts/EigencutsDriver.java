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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.spectral.common.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.common.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.common.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;
import org.apache.mahout.math.stats.OnlineSummarizer;

public class EigencutsDriver extends AbstractJob {

  public static final double EPSILON_DEFAULT = 0.25;

  public static final double TAU_DEFAULT = -0.1;

  public static final double OVERSHOOT_MULTIPLIER = 1.5;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EigencutsDriver(), args);
  }

  @Override
  public int run(String[] arg0) throws Exception {

    // set up command line arguments
    addOption("input", "i", "Path to input affinity matrix data", true);
    addOption("output", "o", "Output of clusterings", true);
    addOption("half-life", "b", "Minimal half-life threshold", true);
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("epsilon", "e", "Half-life threshold coefficient", Double.toString(EPSILON_DEFAULT));
    addOption("tau", "t", "Threshold for cutting affinities", Double.toString(TAU_DEFAULT));
    addOption(DefaultOptionCreator.overwriteOption().create());
    Map<String, String> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    // read in the command line values
    Path input = new Path(parsedArgs.get("--input"));
    Path output = new Path(parsedArgs.get("--output"));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    int dimensions = Integer.parseInt(parsedArgs.get("--dimensions"));
    double halflife = Double.parseDouble(parsedArgs.get("--half-life"));
    double epsilon = Double.parseDouble(parsedArgs.get("--epsilon"));
    double tau = Double.parseDouble(parsedArgs.get("--tau"));

    run(getConf(), input, output, dimensions, halflife, epsilon, tau);

    return 0;
  }

  /**
   * Run the Eigencuts clustering algorithm using the supplied arguments
   * 
   * @param conf the Configuration to use
   * @param input the Path to the directory containing input affinity tuples
   * @param output the Path to the output directory
   * @param dimensions the int number of dimensions of the square affinity matrix
   * @param halflife the double minimum half-life threshold
   * @param epsilon the double coefficient for setting minimum half-life threshold
   * @param tau the double tau threshold for cutting links in the affinity graph
   */
  public static void run(Configuration conf,
                         Path input,
                         Path output,
                         int dimensions,
                         double halflife,
                         double epsilon,
                         double tau)
    throws IOException, InterruptedException, ClassNotFoundException {
    // set the instance variables
    // create a few new Paths for temp files and transformations
    Path outputCalc = new Path(output, "calculations");
    Path outputTmp = new Path(output, "temporary");

    DistributedRowMatrix A = AffinityMatrixInputJob.runJob(input, outputCalc, dimensions);
    Vector D = MatrixDiagonalizeJob.runJob(A.getRowPath(), dimensions);

    long numCuts;
    do {
      // first three steps are the same as spectral k-means:
      // 1) calculate D from A
      // 2) calculate L = D^-0.5 * A * D^-0.5
      // 3) calculate eigenvectors of L

      DistributedRowMatrix L =
          VectorMatrixMultiplicationJob.runJob(A.getRowPath(), D,
              new Path(outputCalc, "laplacian-" + (System.nanoTime() & 0xFF)));
      L.setConf(new Configuration(conf));

      // eigendecomposition (step 3)
      int overshoot = (int) ((double) dimensions * OVERSHOOT_MULTIPLIER);
      List<Double> eigenValues = new ArrayList<Double>(overshoot);
      Matrix eigenVectors = new DenseMatrix(overshoot, dimensions);
      DistributedRowMatrix U = performEigenDecomposition(conf, L, dimensions, overshoot, eigenValues, eigenVectors, outputCalc);
      U.setConf(new Configuration(conf));
      eigenValues = eigenValues.subList(0, dimensions);

      // here's where things get interesting: steps 4, 5, and 6 are unique
      // to this algorithm, and depending on the final output, steps 1-3
      // may be repeated as well

      // helper method, since apparently List and Vector objects don't play nicely
      Vector evs = listToVector(eigenValues);

      // calculate sensitivities (step 4 and step 5)
      Path sensitivities = new Path(outputCalc, "sensitivities-" + (System.nanoTime() & 0xFF));
      EigencutsSensitivityJob.runJob(evs, D, U.getRowPath(), halflife, tau, median(D), epsilon, sensitivities);

      // perform the cuts (step 6)
      input = new Path(outputTmp, "nextAff-" + (System.nanoTime() & 0xFF));
      numCuts = EigencutsAffinityCutsJob.runjob(A.getRowPath(), sensitivities, input, conf);

      // how many cuts were made?
      if (numCuts > 0) {
        // recalculate A
        A = new DistributedRowMatrix(input, new Path(outputTmp, Long.toString(System.nanoTime())), dimensions, dimensions);
        A.setConf(new Configuration());
      }
    } while (numCuts > 0);

    // TODO: MAHOUT-517: Eigencuts needs an output format
  }

  /**
   * Does most of the heavy lifting in setting up Paths, configuring return
   * values, and generally performing the tedious administrative tasks involved
   * in an eigen-decomposition and running the verifier
   */
  public static DistributedRowMatrix performEigenDecomposition(Configuration conf,
                                                               DistributedRowMatrix input,
                                                               int numEigenVectors,
                                                               int overshoot,
                                                               List<Double> eigenValues,
                                                               Matrix eigenVectors, Path tmp) throws IOException {
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    Path seqFiles = new Path(tmp, "eigendecomp-" + (System.nanoTime() & 0xFF));
    solver.runJob(conf,
                  input.getRowPath(),
                  new Path(tmp, "lanczos-" + (System.nanoTime() & 0xFF)),
                  input.numRows(),
                  input.numCols(),
                  true,
                  overshoot,
                  eigenVectors,
                  eigenValues,
                  seqFiles.toString());

    // now run the verifier to trim down the number of eigenvectors
    EigenVerificationJob verifier = new EigenVerificationJob();
    Path verifiedEigens = new Path(tmp, "verifiedeigens");
    verifier.runJob(conf, seqFiles, input.getRowPath(), verifiedEigens, false, 1.0, 0.0, numEigenVectors);
    Path cleanedEigens = verifier.getCleanedEigensPath();
    return new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens, "tmp"), numEigenVectors, input.numRows());
  }

  /**
   * A quick and dirty hack to compute the median of a vector...
   * @param v
   * @return
   */
  private static double median(Vector v) {
    OnlineSummarizer med = new OnlineSummarizer();
    if (v.size() < 100) {
      return v.zSum() / v.size();
    }
    for (Vector.Element e : v) {
      med.add(e.get());
    }
    return med.getMedian();
  }

  /**
   * Iteratively loops through the list, converting it to a Vector of double
   * primitives worthy of other Mahout operations
   */
  private static Vector listToVector(Collection<Double> list) {
    Vector retval = new DenseVector(list.size());
    int index = 0;
    for (Double d : list) {
      retval.setQuick(index++, d);
    }
    return retval;
  }

}
