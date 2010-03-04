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

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class DistributedLanczosSolver extends LanczosSolver implements Tool {

  private static final Logger log = LoggerFactory.getLogger(DistributedLanczosSolver.class);

  private Configuration conf;

  private Map<String,String> parsedArgs;

  /**
   * For the distributed case, the best guess at a useful initialization state for Lanczos we'll chose to be
   * uniform over all input dimensions, L_2 normalized.
   * @param corpus
   * @return
   */
  @Override
  protected Vector getInitialVector(VectorIterable corpus) {
    Vector initialVector = new DenseVector(corpus.numCols());
    initialVector.assign(1/Math.sqrt(corpus.numCols()));
    return initialVector;
  }

  @Override
  public int run(String[] strings) throws Exception {
    String inputPathString = parsedArgs.get("--input");
    String outputTmpPathString = parsedArgs.get("--tempDir");
    int numRows = Integer.parseInt(parsedArgs.get("--numRows"));
    int numCols = Integer.parseInt(parsedArgs.get("--numCols"));
    int desiredRank = Integer.parseInt(parsedArgs.get("--rank"));
    Matrix eigenVectors = new DenseMatrix(desiredRank, numCols);
    List<Double> eigenValues = new ArrayList<Double>();
    String outputEigenVectorPath =  parsedArgs.get("--output");
    
    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPathString,
                                                           outputTmpPathString,
                                                           numRows,
                                                           numCols);
    matrix.configure(new JobConf(getConf()));
    solve(matrix, desiredRank, eigenVectors, eigenValues);

    serializeOutput(eigenVectors, eigenValues, outputEigenVectorPath);  
    return 0;
  }

  /**
   * TODO: this should be refactored to allow both LanczosSolver impls to properly serialize output in a generic way.
   * @param eigenVectors The eigenvectors to be serialized
   * @param eigenValues The eigenvalues to be serialized
   * @param outputPath The path (relative to the current Configuration's FileSystem) to save the output to.
   * @throws IOException
   */
  public void serializeOutput(Matrix eigenVectors, List<Double> eigenValues, String outputPath) throws IOException {
    log.info("Persisting " + eigenVectors.numRows() + " eigenVectors and eigenValues to: " + outputPath);
    Path path = new Path(outputPath);
    Configuration conf = getConf();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    VectorWritable vw = new VectorWritable();
    IntWritable iw = new IntWritable();
    for(int i=0; i<eigenVectors.numRows() - 1; i++) {
      Vector v = eigenVectors.getRow(i);
      v.setName("eigenVector" + i + ", eigenvalue = " + eigenValues.get(i));
      vw.set(v);
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

      Option numRowsOpt = buildOption("numRows",
                                      "nr",
                                      "Number of rows of the input matrix");
      Option numColsOpt = buildOption("numCols",
                                      "nc",
                                      "Number of columns of the input matrix");
      Option desiredRankOpt = buildOption("rank",
                                          "r",
                                          "Desired decomposition rank (note: only roughly 1/4 to 1/3 "
                                        + "of these will have the top portion of the spectrum)");

      DistributedLanczosSolver.this.parsedArgs = parseArguments(args, numRowsOpt, numColsOpt, desiredRankOpt);
      return DistributedLanczosSolver.this.run(args);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistributedLanczosSolver().job(), args);
  }
}
