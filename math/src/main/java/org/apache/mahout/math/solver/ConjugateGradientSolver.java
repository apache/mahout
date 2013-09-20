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

package org.apache.mahout.math.solver;

import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Implementation of a conjugate gradient iterative solver for linear systems. Implements both
 * standard conjugate gradient and pre-conditioned conjugate gradient. 
 * 
 * <p>Conjugate gradient requires the matrix A in the linear system Ax = b to be symmetric and positive
 * definite. For convenience, this implementation allows the input matrix to be be non-symmetric, in
 * which case the system A'Ax = b is solved. Because this requires only one pass through the matrix A, it
 * is faster than explicitly computing A'A, then passing the results to the solver.
 * 
 * <p>For inputs that may be ill conditioned (often the case for highly sparse input), this solver
 * also accepts a parameter, lambda, which adds a scaled identity to the matrix A, solving the system
 * (A + lambda*I)x = b. This obviously changes the solution, but it will guarantee solvability. The
 * ridge regression approach to linear regression is a common use of this feature.
 * 
 * <p>If only an approximate solution is required, the maximum number of iterations or the error threshold
 * may be specified to end the algorithm early at the expense of accuracy. When the matrix A is ill conditioned,
 * it may sometimes be necessary to increase the maximum number of iterations above the default of A.numCols()
 * due to numerical issues.
 * 
 * <p>By default the solver will run a.numCols() iterations or until the residual falls below 1E-9.
 * 
 * <p>For more information on the conjugate gradient algorithm, see Golub & van Loan, "Matrix Computations", 
 * sections 10.2 and 10.3 or the <a href="http://en.wikipedia.org/wiki/Conjugate_gradient">conjugate gradient
 * wikipedia article</a>.
 */

public class ConjugateGradientSolver {

  public static final double DEFAULT_MAX_ERROR = 1.0e-9;
  
  private static final Logger log = LoggerFactory.getLogger(ConjugateGradientSolver.class);
  private static final PlusMult PLUS_MULT = new PlusMult(1.0);

  private int iterations;
  private double residualNormSquared;
  
  public ConjugateGradientSolver() {
    this.iterations = 0;
    this.residualNormSquared = Double.NaN;
  }  

  /**
   * Solves the system Ax = b with default termination criteria. A must be symmetric, square, and positive definite.
   * Only the squareness of a is checked, since testing for symmetry and positive definiteness are too expensive. If
   * an invalid matrix is specified, then the algorithm may not yield a valid result.
   *  
   * @param a  The linear operator A.
   * @param b  The vector b.
   * @return The result x of solving the system.
   * @throws IllegalArgumentException if a is not square or if the size of b is not equal to the number of columns of a.
   * 
   */
  public Vector solve(VectorIterable a, Vector b) {
    return solve(a, b, null, b.size(), DEFAULT_MAX_ERROR);
  }
  
  /**
   * Solves the system Ax = b with default termination criteria using the specified preconditioner. A must be 
   * symmetric, square, and positive definite. Only the squareness of a is checked, since testing for symmetry 
   * and positive definiteness are too expensive. If an invalid matrix is specified, then the algorithm may not 
   * yield a valid result.
   *  
   * @param a  The linear operator A.
   * @param b  The vector b.
   * @param precond A preconditioner to use on A during the solution process.
   * @return The result x of solving the system.
   * @throws IllegalArgumentException if a is not square or if the size of b is not equal to the number of columns of a.
   * 
   */
  public Vector solve(VectorIterable a, Vector b, Preconditioner precond) {
    return solve(a, b, precond, b.size(), DEFAULT_MAX_ERROR);
  }
  

  /**
   * Solves the system Ax = b, where A is a linear operator and b is a vector. Uses the specified preconditioner
   * to improve numeric stability and possibly speed convergence. This version of solve() allows control over the 
   * termination and iteration parameters.
   * 
   * @param a  The matrix A.
   * @param b  The vector b.
   * @param preconditioner The preconditioner to apply.
   * @param maxIterations The maximum number of iterations to run.
   * @param maxError The maximum amount of residual error to tolerate. The algorithm will run until the residual falls 
   * below this value or until maxIterations are completed.
   * @return The result x of solving the system.
   * @throws IllegalArgumentException if the matrix is not square, if the size of b is not equal to the number of 
   * columns of A, if maxError is less than zero, or if maxIterations is not positive. 
   */
  
  public Vector solve(VectorIterable a, 
                      Vector b, 
                      Preconditioner preconditioner, 
                      int maxIterations, 
                      double maxError) {

    if (a.numRows() != a.numCols()) {
      throw new IllegalArgumentException("Matrix must be square, symmetric and positive definite.");
    }
    
    if (a.numCols() != b.size()) {
      throw new CardinalityException(a.numCols(), b.size());
    }

    if (maxIterations <= 0) {
      throw new IllegalArgumentException("Max iterations must be positive.");      
    }
    
    if (maxError < 0.0) {
      throw new IllegalArgumentException("Max error must be non-negative.");
    }
    
    Vector x = new DenseVector(b.size());

    iterations = 0;
    Vector residual = b.minus(a.times(x));
    residualNormSquared = residual.dot(residual);

    log.info("Conjugate gradient initial residual norm = {}", Math.sqrt(residualNormSquared));
    double previousConditionedNormSqr = 0.0;
    Vector updateDirection = null;
    while (Math.sqrt(residualNormSquared) > maxError && iterations < maxIterations) {
      Vector conditionedResidual;
      double conditionedNormSqr;
      if (preconditioner == null) {
        conditionedResidual = residual;
        conditionedNormSqr = residualNormSquared;
      } else {
        conditionedResidual = preconditioner.precondition(residual);
        conditionedNormSqr = residual.dot(conditionedResidual);
      }      
      
      ++iterations;
      
      if (iterations == 1) {
        updateDirection = new DenseVector(conditionedResidual);
      } else {
        double beta = conditionedNormSqr / previousConditionedNormSqr;
        
        // updateDirection = residual + beta * updateDirection
        updateDirection.assign(Functions.MULT, beta);
        updateDirection.assign(conditionedResidual, Functions.PLUS);
      }
      
      Vector aTimesUpdate = a.times(updateDirection);
      
      double alpha = conditionedNormSqr / updateDirection.dot(aTimesUpdate);
      
      // x = x + alpha * updateDirection
      PLUS_MULT.setMultiplicator(alpha);
      x.assign(updateDirection, PLUS_MULT);

      // residual = residual - alpha * A * updateDirection
      PLUS_MULT.setMultiplicator(-alpha);
      residual.assign(aTimesUpdate, PLUS_MULT);
      
      previousConditionedNormSqr = conditionedNormSqr;
      residualNormSquared = residual.dot(residual);
      
      log.info("Conjugate gradient iteration {} residual norm = {}", iterations, Math.sqrt(residualNormSquared));
    }
    return x;
  }

  /**
   * Returns the number of iterations run once the solver is complete.
   * 
   * @return The number of iterations run.
   */
  public int getIterations() {
    return iterations;
  }

  /**
   * Returns the norm of the residual at the completion of the solver. Usually this should be close to zero except in
   * the case of a non positive definite matrix A, which results in an unsolvable system, or for ill conditioned A, in
   * which case more iterations than the default may be needed.
   * 
   * @return The norm of the residual in the solution.
   */
  public double getResidualNorm() {
    return Math.sqrt(residualNormSquared);
  }  
}
