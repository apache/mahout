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

package org.apache.mahout.math.hadoop.decomposer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.decomposer.lanczos.LanczosSolver;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DistributedLanczosSolver extends LanczosSolver implements Tool {

  public static final String RAW_EIGENVECTORS = "rawEigenvectors";

  private static final Logger log = LoggerFactory.getLogger(DistributedLanczosSolver.class);

  private Configuration conf;

  private Map<String, String> parsedArgs;

  /**
   * For the distributed case, the best guess at a useful initialization state for Lanczos we'll chose to be
   * uniform over all input dimensions, L_2 normalized.
   */
  @Override
  protected Vector getInitialVector(VectorIterable corpus) {
    Vector initialVector = new DenseVector(corpus.numCols());
    initialVector.assign(1.0 / Math.sqrt(corpus.numCols()));
    return initialVector;
  }
  
  /**
   * Factored-out LanczosSolver for the purpose of invoking it programmatically
   */
  public void runJob(Configuration originalConfig,
                     Path inputPath,
                     Path outputTmpPath,
                     int numRows,
                     int numCols,
                     boolean isSymmetric,
                     int desiredRank,
                     Matrix eigenVectors,
                     List<Double> eigenValues,
                     String outputEigenVectorPathString) throws IOException {
	  DistributedRowMatrix matrix =
        new DistributedRowMatrix(inputPath, outputTmpPath, numRows, numCols);
	  matrix.setConf(new Configuration(originalConfig));
	  setConf(originalConfig);
	  solve(matrix, desiredRank, eigenVectors, eigenValues, isSymmetric);
	  serializeOutput(eigenVectors, eigenValues, new Path(outputEigenVectorPathString));
  }

  @Override
  public int run(String[] strings) throws Exception {
    Path inputPath = new Path(parsedArgs.get("--input"));
    Path outputPath = new Path(parsedArgs.get("--output"));
    Path outputTmpPath = new Path(parsedArgs.get("--tempDir"));
    int numRows = Integer.parseInt(parsedArgs.get("--numRows"));
    int numCols = Integer.parseInt(parsedArgs.get("--numCols"));
    boolean isSymmetric = Boolean.parseBoolean(parsedArgs.get("--symmetric"));
    int desiredRank = Integer.parseInt(parsedArgs.get("--rank"));

    boolean cleansvd = Boolean.parseBoolean(parsedArgs.get("--cleansvd"));
    if (cleansvd) {
      double maxError = Double.parseDouble(parsedArgs.get("--maxError"));
      double minEigenvalue = Double.parseDouble(parsedArgs.get("--minEigenvalue"));
      boolean inMemory = Boolean.parseBoolean(parsedArgs.get("--inMemory"));
      return run(inputPath,
                 outputPath,
                 outputTmpPath,
                 numRows,
                 numCols,
                 isSymmetric,
                 desiredRank,
                 maxError,
                 minEigenvalue,
                 inMemory);
    }
    return run(inputPath, outputPath, outputTmpPath, numRows, numCols, isSymmetric, desiredRank);
  }

  /**
   * Run the solver to produce raw eigenvectors, then run the EigenVerificationJob to clean them
   * 
   * @param inputPath the Path to the input corpus
   * @param outputPath the Path to the output
   * @param outputTmpPath a Path to a temporary working directory
   * @param numRows the int number of rows 
   * @param numCols the int number of columns
   * @param isSymmetric true if the input matrix is symmetric
   * @param desiredRank the int desired rank of eigenvectors to produce
   * @param maxError the maximum allowable error
   * @param minEigenvalue the minimum usable eigenvalue
   * @param inMemory true if the verification can be done in memory
   * @return an int indicating success (0) or otherwise
   */
  public int run(Path inputPath,
                 Path outputPath,
                 Path outputTmpPath,
                 int numRows,
                 int numCols,
                 boolean isSymmetric,
                 int desiredRank,
                 double maxError,
                 double minEigenvalue,
                 boolean inMemory) throws Exception {
    int result = run(inputPath, outputPath, outputTmpPath, numRows, numCols, isSymmetric, desiredRank);
    if (result != 0) {
      return result;
    }
    Path rawEigenVectorPath = new Path(outputPath, RAW_EIGENVECTORS);
    return new EigenVerificationJob().run(inputPath,
                                          rawEigenVectorPath,
                                          outputPath,
                                          outputTmpPath,
                                          maxError,
                                          minEigenvalue,
                                          inMemory,
                                          getConf() != null ? new Configuration(getConf()) : new Configuration());
  }

  /**
   * Run the solver to produce the raw eigenvectors
   * 
   * @param inputPath the Path to the input corpus
   * @param outputPath the Path to the output
   * @param outputTmpPath a Path to a temporary working directory
   * @param numRows the int number of rows 
   * @param numCols the int number of columns
   * @param isSymmetric true if the input matrix is symmetric
   * @param desiredRank the int desired rank of eigenvectors to produce
   * @return  an int indicating success (0) or otherwise
   */
  public int run(Path inputPath,
                 Path outputPath,
                 Path outputTmpPath,
                 int numRows,
                 int numCols,
                 boolean isSymmetric,
                 int desiredRank) throws Exception {
    Matrix eigenVectors = new DenseMatrix(desiredRank, numCols);
    List<Double> eigenValues = new ArrayList<Double>();

    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPath, outputTmpPath, numRows, numCols);
    matrix.setConf(new Configuration(getConf() != null ? getConf() : new Configuration()));
    solve(matrix, desiredRank, eigenVectors, eigenValues, isSymmetric);

    Path outputEigenVectorPath = new Path(outputPath, RAW_EIGENVECTORS);
    serializeOutput(eigenVectors, eigenValues, outputEigenVectorPath);
    return 0;
  }

  /**
   * @param eigenVectors The eigenvectors to be serialized
   * @param eigenValues The eigenvalues to be serialized
   * @param outputPath The path (relative to the current Configuration's FileSystem) to save the output to.
   */
  public void serializeOutput(Matrix eigenVectors, List<Double> eigenValues, Path outputPath) throws IOException {
    log.info("Persisting {} eigenVectors and eigenValues to: {}", eigenVectors.numRows(), outputPath);
    Configuration conf = getConf() != null ? getConf() : new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter =
        new SequenceFile.Writer(fs, conf, outputPath, IntWritable.class, VectorWritable.class);
    IntWritable iw = new IntWritable();
    for (int i = 0; i < eigenVectors.numRows() - 1; i++) {
      Vector v = eigenVectors.getRow(i);
      Writable vw = new VectorWritable(v);
      iw.set(i);
      seqWriter.append(iw, vw);
    }
    seqWriter.close();
  }

  @Override
  public void setConf(Configuration configuration) {
    conf = configuration;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  public DistributedLanczosSolverJob job() {
    return new DistributedLanczosSolverJob();
  }

  /**
   * Inner subclass of AbstractJob so we get access to AbstractJob's functionality w.r.t. cmdline options, but still
   * sublcass LanczosSolver.
   */
  public class DistributedLanczosSolverJob extends AbstractJob {
    @Override
    public void setConf(Configuration conf) {
      DistributedLanczosSolver.this.setConf(conf);
    }

    @Override
    public Configuration getConf() {
      return DistributedLanczosSolver.this.getConf();
    }

    @Override
    public int run(String[] args) throws Exception {
      addInputOption();
      addOutputOption();
      addOption("numRows", "nr", "Number of rows of the input matrix");
      addOption("numCols", "nc", "Number of columns of the input matrix");
      addOption("rank", "r", "Desired decomposition rank (note: only roughly 1/4 to 1/3 "
          + "of these will have the top portion of the spectrum)");
      addOption("symmetric", "sym", "Is the input matrix square and symmetric?");
      // options required to run cleansvd job
      addOption("cleansvd", "cl", "Run the EigenVerificationJob to clean the eigenvectors after SVD", false);
      addOption("maxError", "err", "Maximum acceptable error", "0.05");
      addOption("minEigenvalue", "mev", "Minimum eigenvalue to keep the vector for", "0.0");
      addOption("inMemory", "mem", "Buffer eigen matrix into memory (if you have enough!)", "false");

      DistributedLanczosSolver.this.parsedArgs = parseArguments(args);
      if (DistributedLanczosSolver.this.parsedArgs == null) {
        return -1;
      } else {
        return DistributedLanczosSolver.this.run(args);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistributedLanczosSolver().job(), args);
  }
}
