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

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.EigenSeedGenerator;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.spectral.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.UnitVectorizerJob;
import org.apache.mahout.clustering.spectral.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDSolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Performs spectral k-means clustering on the top k eigenvectors of the input affinity matrix.
 */
public class SpectralKMeansDriver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(SpectralKMeansDriver.class);

  public static final double OVERSHOOTMULTIPLIER = 2.0;
  public static final int REDUCERS = 10;
  public static final int BLOCKHEIGHT = 30000;
  public static final int OVERSAMPLING = 15;
  public static final int POWERITERS = 0;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SpectralKMeansDriver(), args);
  }

  @Override
  public int run(String[] arg0) throws Exception {

    Configuration conf = getConf();
    addInputOption();
    addOutputOption();
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("clusters", "k", "Number of clusters and top eigenvectors", true);
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addFlag("usessvd", "ssvd", "Uses SSVD as the eigensolver. Default is the Lanczos solver.");
    addOption("reduceTasks", "t", "Number of reducers for SSVD", String.valueOf(REDUCERS));
    addOption("outerProdBlockHeight", "oh", "Block height of outer products for SSVD", String.valueOf(BLOCKHEIGHT));
    addOption("oversampling", "p", "Oversampling parameter for SSVD", String.valueOf(OVERSAMPLING));
    addOption("powerIter", "q", "Additional power iterations for SSVD", String.valueOf(POWERITERS));

    Map<String,List<String>> parsedArgs = parseArguments(arg0);
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

    Path tempdir = new Path(getOption("tempDir"));
    boolean ssvd = parsedArgs.containsKey("--usessvd");
    if (ssvd) {
      int reducers = Integer.parseInt(getOption("reduceTasks"));
      int blockheight = Integer.parseInt(getOption("outerProdBlockHeight"));
      int oversampling = Integer.parseInt(getOption("oversampling"));
      int poweriters = Integer.parseInt(getOption("powerIter"));
      run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations, tempdir, true, reducers,
          blockheight, oversampling, poweriters);
    } else {
      run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations, tempdir, false);
    }

    return 0;
  }

  public static void run(Configuration conf, Path input, Path output, int numDims, int clusters,
      DistanceMeasure measure, double convergenceDelta, int maxIterations, Path tempDir, boolean ssvd)
      throws IOException, InterruptedException, ClassNotFoundException {
    run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations, tempDir, ssvd, REDUCERS,
        BLOCKHEIGHT, OVERSAMPLING, POWERITERS);
  }

  /**
   * Run the Spectral KMeans clustering on the supplied arguments
   *
   * @param conf
   *          the Configuration to be used
   * @param input
   *          the Path to the input tuples directory
   * @param output
   *          the Path to the output directory
   * @param numDims
   *          the int number of dimensions of the affinity matrix
   * @param clusters
   *          the int number of eigenvectors and thus clusters to produce
   * @param measure
   *          the DistanceMeasure for the k-Means calculations
   * @param convergenceDelta
   *          the double convergence delta for the k-Means calculations
   * @param maxIterations
   *          the int maximum number of iterations for the k-Means calculations
   * @param tempDir
   *          Temporary directory for intermediate calculations
   * @param ssvd
   *          Flag to indicate the eigensolver to use
   * @param numReducers
   *          Number of reducers
   * @param blockHeight
   * @param oversampling
   * @param poweriters
   */
  public static void run(Configuration conf, Path input, Path output, int numDims, int clusters,
      DistanceMeasure measure, double convergenceDelta, int maxIterations, Path tempDir, boolean ssvd, int numReducers,
      int blockHeight, int oversampling, int poweriters) throws IOException, InterruptedException,
      ClassNotFoundException {

    Path outputCalc = new Path(tempDir, "calculations");
    Path outputTmp = new Path(tempDir, "temporary");

    // Take in the raw CSV text file and split it ourselves,
    // creating our own SequenceFiles for the matrices to read later
    // (similar to the style of syntheticcontrol.canopy.InputMapper)
    Path affSeqFiles = new Path(outputCalc, "seqfile");
    AffinityMatrixInputJob.runJob(input, affSeqFiles, numDims, numDims);

    // Construct the affinity matrix using the newly-created sequence files
    DistributedRowMatrix A = new DistributedRowMatrix(affSeqFiles, new Path(outputTmp, "afftmp"), numDims, numDims);

    Configuration depConf = new Configuration(conf);
    A.setConf(depConf);

    // Construct the diagonal matrix D (represented as a vector)
    Vector D = MatrixDiagonalizeJob.runJob(affSeqFiles, numDims);

    // Calculate the normalized Laplacian of the form: L = D^(-0.5)AD^(-0.5)
    DistributedRowMatrix L = VectorMatrixMultiplicationJob.runJob(affSeqFiles, D, new Path(outputCalc, "laplacian"),
        new Path(outputCalc, outputCalc));
    L.setConf(depConf);

    Path data;

    if (ssvd) {
      // SSVD requires an array of Paths to function. So we pass in an array of length one
      Path[] LPath = new Path[1];
      LPath[0] = L.getRowPath();

      Path SSVDout = new Path(outputCalc, "SSVD");

      SSVDSolver solveIt = new SSVDSolver(depConf, LPath, SSVDout, blockHeight, clusters, oversampling, numReducers);

      solveIt.setComputeV(false);
      solveIt.setComputeU(true);
      solveIt.setOverwrite(true);
      solveIt.setQ(poweriters);
      // solveIt.setBroadcast(false);
      solveIt.run();
      data = new Path(solveIt.getUPath());
    } else {
      // Perform eigen-decomposition using LanczosSolver
      // since some of the eigen-output is spurious and will be eliminated
      // upon verification, we have to aim to overshoot and then discard
      // unnecessary vectors later
      int overshoot = Math.min((int) (clusters * OVERSHOOTMULTIPLIER), numDims);
      DistributedLanczosSolver solver = new DistributedLanczosSolver();
      LanczosState state = new LanczosState(L, overshoot, DistributedLanczosSolver.getInitialVector(L));
      Path lanczosSeqFiles = new Path(outputCalc, "eigenvectors");

      solver.runJob(conf, state, overshoot, true, lanczosSeqFiles.toString());

      // perform a verification
      EigenVerificationJob verifier = new EigenVerificationJob();
      Path verifiedEigensPath = new Path(outputCalc, "eigenverifier");
      verifier.runJob(conf, lanczosSeqFiles, L.getRowPath(), verifiedEigensPath, true, 1.0, clusters);

      Path cleanedEigens = verifier.getCleanedEigensPath();
      DistributedRowMatrix W = new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens, "tmp"), clusters,
          numDims);
      W.setConf(depConf);
      DistributedRowMatrix Wtrans = W.transpose();
      data = Wtrans.getRowPath();
    }

    // Normalize the rows of Wt to unit length
    // normalize is important because it reduces the occurrence of two unique clusters combining into one
    Path unitVectors = new Path(outputCalc, "unitvectors");

    UnitVectorizerJob.runJob(data, unitVectors);

    DistributedRowMatrix Wt = new DistributedRowMatrix(unitVectors, new Path(unitVectors, "tmp"), clusters, numDims);
    Wt.setConf(depConf);
    data = Wt.getRowPath();

    // Generate initial clusters using EigenSeedGenerator which picks rows as centroids if that row contains max
    // eigen value in that column
    Path initialclusters = EigenSeedGenerator.buildFromEigens(conf, data,
        new Path(output, Cluster.INITIAL_CLUSTERS_DIR), clusters, measure);

    // Run the KMeansDriver
    Path answer = new Path(output, "kmeans_out");
    KMeansDriver.run(conf, data, initialclusters, answer, convergenceDelta, maxIterations, true, 0.0, false);

    // Restore name to id mapping and read through the cluster assignments
    Path mappingPath = new Path(new Path(conf.get("hadoop.tmp.dir")), "generic_input_mapping");
    List<String> mapping = Lists.newArrayList();
    FileSystem fs = FileSystem.get(mappingPath.toUri(), conf);
    if (fs.exists(mappingPath)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, mappingPath, conf);
      Text mappingValue = new Text();
      IntWritable mappingIndex = new IntWritable();
      while (reader.next(mappingIndex, mappingValue)) {
        String s = mappingValue.toString();
        mapping.add(s);
      }
      HadoopUtil.delete(conf, mappingPath);
    } else {
      log.warn("generic input mapping file not found!");
    }

    Path clusteredPointsPath = new Path(answer, "clusteredPoints");
    Path inputPath = new Path(clusteredPointsPath, "part-m-00000");
    int id = 0;
    for (Pair<IntWritable, WeightedVectorWritable> record :
         new SequenceFileIterable<IntWritable, WeightedVectorWritable>(inputPath, conf)) {
      if (!mapping.isEmpty()) {
        log.info("{}: {}", mapping.get(id++), record.getFirst().get());
      } else {
        log.info("{}: {}", id++, record.getFirst().get());
      }
    }
  }
}
