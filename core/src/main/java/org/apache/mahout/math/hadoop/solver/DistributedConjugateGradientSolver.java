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

package org.apache.mahout.math.hadoop.solver;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.solver.ConjugateGradientSolver;
import org.apache.mahout.math.solver.Preconditioner;

/**
 * Distributed implementation of the conjugate gradient solver. More or less, this is just the standard solver
 * but wrapped with some methods that make it easy to run it on a DistributedRowMatrix.
 */
public class DistributedConjugateGradientSolver extends ConjugateGradientSolver implements Tool {

  private Configuration conf; 
  private Map<String, List<String>> parsedArgs;

  /**
   * 
   * Runs the distributed conjugate gradient solver programmatically to solve the system (A + lambda*I)x = b.
   * 
   * @param inputPath      Path to the matrix A
   * @param tempPath       Path to scratch output path, deleted after the solver completes
   * @param numRows        Number of rows in A
   * @param numCols        Number of columns in A
   * @param b              Vector b
   * @param preconditioner Optional preconditioner for the system
   * @param maxIterations  Maximum number of iterations to run, defaults to numCols
   * @param maxError       Maximum error tolerated in the result. If the norm of the residual falls below this,
   *                       then the algorithm stops and returns.
   * @return               The vector that solves the system.
   */
  public Vector runJob(Path inputPath, 
                       Path tempPath, 
                       int numRows, 
                       int numCols, 
                       Vector b, 
                       Preconditioner preconditioner, 
                       int maxIterations, 
                       double maxError) {
    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPath, tempPath, numRows, numCols);
    matrix.setConf(conf);
        
    return solve(matrix, b, preconditioner, maxIterations, maxError);
  }
  
  @Override
  public Configuration getConf() {
    return conf;
  }

  @Override
  public void setConf(Configuration conf) {
    this.conf = conf;    
  }

  @Override
  public int run(String[] strings) throws Exception {
    Path inputPath = new Path(AbstractJob.getOption(parsedArgs, "--input"));
    Path outputPath = new Path(AbstractJob.getOption(parsedArgs, "--output"));
    Path tempPath = new Path(AbstractJob.getOption(parsedArgs, "--tempDir"));
    Path vectorPath = new Path(AbstractJob.getOption(parsedArgs, "--vector"));
    int numRows = Integer.parseInt(AbstractJob.getOption(parsedArgs, "--numRows"));
    int numCols = Integer.parseInt(AbstractJob.getOption(parsedArgs, "--numCols"));
    int maxIterations = parsedArgs.containsKey("--maxIter")
        ? Integer.parseInt(AbstractJob.getOption(parsedArgs, "--maxIter"))
        : numCols;
    double maxError = parsedArgs.containsKey("--maxError") 
        ? Double.parseDouble(AbstractJob.getOption(parsedArgs, "--maxError"))
        : ConjugateGradientSolver.DEFAULT_MAX_ERROR;

    Vector b = loadInputVector(vectorPath);
    Vector x = runJob(inputPath, tempPath, numRows, numCols, b, null, maxIterations, maxError);
    saveOutputVector(outputPath, x);
    tempPath.getFileSystem(conf).delete(tempPath, true);
    
    return 0;
  }
  
  public DistributedConjugateGradientSolverJob job() {
    return new DistributedConjugateGradientSolverJob();
  }
  
  private Vector loadInputVector(Path path) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    try {
      VectorWritable value = new VectorWritable();
      if (!reader.next(new IntWritable(), value)) {
        throw new IOException("Input vector file is empty.");      
      }
      return value.get();
    } finally {
      reader.close();
    }
  }
  
  private void saveOutputVector(Path path, Vector v) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    
    try {
      writer.append(new IntWritable(0), new VectorWritable(v));
    } finally {
      writer.close();
    }
  }
  
  public class DistributedConjugateGradientSolverJob extends AbstractJob {
    @Override
    public void setConf(Configuration conf) {
      DistributedConjugateGradientSolver.this.setConf(conf);
    }
    
    @Override
    public Configuration getConf() {
      return DistributedConjugateGradientSolver.this.getConf();
    }
    
    @Override
    public int run(String[] args) throws Exception {
      addInputOption();
      addOutputOption();
      addOption("numRows", "nr", "Number of rows in the input matrix", true);
      addOption("numCols", "nc", "Number of columns in the input matrix", true);
      addOption("vector", "b", "Vector to solve against", true);
      addOption("lambda", "l", "Scalar in A + lambda * I [default = 0]", "0.0");
      addOption("symmetric", "sym", "Is the input matrix square and symmetric?", "true");
      addOption("maxIter", "x", "Maximum number of iterations to run");
      addOption("maxError", "err", "Maximum residual error to allow before stopping");

      DistributedConjugateGradientSolver.this.parsedArgs = parseArguments(args);
      if (DistributedConjugateGradientSolver.this.parsedArgs == null) {
        return -1;
      } else {
        Configuration conf = getConf();
        if (conf == null) {
          conf = new Configuration();
        }
        DistributedConjugateGradientSolver.this.setConf(conf);
        return DistributedConjugateGradientSolver.this.run(args);
      }
    }    
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DistributedConjugateGradientSolver().job(), args);
  }
}
