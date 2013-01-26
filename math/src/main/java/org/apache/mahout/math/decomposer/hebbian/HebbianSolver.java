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

package org.apache.mahout.math.decomposer.hebbian;

import java.util.List;
import java.util.Properties;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.decomposer.AsyncEigenVerifier;
import org.apache.mahout.math.decomposer.EigenStatus;
import org.apache.mahout.math.decomposer.SingularVectorVerifier;
import org.apache.mahout.math.function.TimesFunction;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.PlusMult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The Hebbian solver is an iterative, sparse, singular value decomposition solver, based on the paper
 * <a href="http://www.dcs.shef.ac.uk/~genevieve/gorrell_webb.pdf">Generalized Hebbian Algorithm for
 * Latent Semantic Analysis</a> (2005) by Genevieve Gorrell and Brandyn Webb (a.k.a. Simon Funk).
 * TODO: more description here!  For now: read the inline comments, and the comments for the constructors.
 */
public class HebbianSolver {

  private static final Logger log = LoggerFactory.getLogger(HebbianSolver.class);
  private static final boolean DEBUG = false;

  private final EigenUpdater updater;
  private final SingularVectorVerifier verifier;
  private final double convergenceTarget;
  private final int maxPassesPerEigen;
  private final Random rng = RandomUtils.getRandom();

  private int numPasses = 0;

  /**
   * Creates a new HebbianSolver
   *
   * @param updater
   *  {@link EigenUpdater} used to do the actual work of iteratively updating the current "best guess"
   *   singular vector one data-point presentation at a time.
   * @param verifier
   *  {@link SingularVectorVerifier } an object which perpetually tries to check how close to
   *   convergence the current singular vector is (typically is a
   *  {@link org.apache.mahout.math.decomposer.AsyncEigenVerifier } which does this
   *   in the background in another thread, while the main thread continues to converge)
   * @param convergenceTarget a small "epsilon" value which tells the solver how small you want the cosine of the
   *  angle between a proposed eigenvector and that same vector after being multiplied by the (square of the) input
   *  corpus
   * @param maxPassesPerEigen a cutoff which tells the solver after how many times of checking for convergence (done
   *  by the verifier) should the solver stop trying, even if it has not reached the convergenceTarget.
   */
  public HebbianSolver(EigenUpdater updater,
                       SingularVectorVerifier verifier,
                       double convergenceTarget,
                       int maxPassesPerEigen) {
    this.updater = updater;
    this.verifier = verifier;
    this.convergenceTarget = convergenceTarget;
    this.maxPassesPerEigen = maxPassesPerEigen;
  }

  /**
   * Creates a new HebbianSolver with maxPassesPerEigen = Integer.MAX_VALUE (i.e. keep on iterating until
   * convergenceTarget is reached).  <b>Not recommended</b> unless only looking for
   * the first few (5, maybe 10?) singular
   * vectors, as small errors which compound early on quickly put a minimum error on subsequent vectors.
   *
   * @param updater {@link EigenUpdater} used to do the actual work of iteratively updating the current "best guess"
   *  singular vector one data-point presentation at a time.
   * @param verifier {@link org.apache.mahout.math.decomposer.SingularVectorVerifier }
   * an object which perpetually tries to check how close to
   *  convergence the current singular vector is (typically is a
   * {@link org.apache.mahout.math.decomposer.AsyncEigenVerifier } which does this
   *  in the background in another thread, while the main thread continues to converge)
   * @param convergenceTarget a small "epsilon" value which tells the solver how small you want the cosine of the
   *  angle between a proposed eigenvector and that same vector after being multiplied by the (square of the) input
   *  corpus
   */
  public HebbianSolver(EigenUpdater updater,
                       SingularVectorVerifier verifier,
                       double convergenceTarget) {
    this(updater,
        verifier,
        convergenceTarget,
        Integer.MAX_VALUE);
  }

  /**
   * <b>This is the recommended constructor to use if you're not sure</b>
   * Creates a new HebbianSolver with the default {@link HebbianUpdater } to do the updating work, and the default
   * {@link org.apache.mahout.math.decomposer.AsyncEigenVerifier } to check for convergence in a
   * (single) background thread.
   *
   * @param convergenceTarget a small "epsilon" value which tells the solver how small you want the cosine of the
   *  angle between a proposed eigenvector and that same vector after being multiplied by the (square of the) input
   *  corpus
   * @param maxPassesPerEigen a cutoff which tells the solver after how many times of checking for convergence (done
   *  by the verifier) should the solver stop trying, even if it has not reached the convergenceTarget.
   */
  public HebbianSolver(double convergenceTarget, int maxPassesPerEigen) {
    this(new HebbianUpdater(),
        new AsyncEigenVerifier(),
        convergenceTarget,
        maxPassesPerEigen);
  }

  /**
   * Creates a new HebbianSolver with the default {@link HebbianUpdater } to do the updating work, and the default
   * {@link org.apache.mahout.math.decomposer.AsyncEigenVerifier } to check for convergence in a (single)
   * background thread, with
   * maxPassesPerEigen set to Integer.MAX_VALUE.  <b>Not recommended</b> unless only looking
   * for the first few (5, maybe 10?) singular
   * vectors, as small errors which compound early on quickly put a minimum error on subsequent vectors.
   *
   * @param convergenceTarget a small "epsilon" value which tells the solver how small you want the cosine of the
   *  angle between a proposed eigenvector and that same vector after being multiplied by the (square of the) input
   *  corpus
   */
  public HebbianSolver(double convergenceTarget) {
    this(convergenceTarget, Integer.MAX_VALUE);
  }

  /**
   * Creates a new HebbianSolver with the default {@link HebbianUpdater } to do the updating work, and the default
   * {@link org.apache.mahout.math.decomposer.AsyncEigenVerifier } to check for convergence in a (single)
   * background thread, with
   * convergenceTarget set to 0, which means that the solver will not really care about convergence as a loop-exiting
   * criterion (but will be checking for convergence anyways, so it will be logged and singular values will be
   * saved).
   *
   * @param numPassesPerEigen the exact number of times the verifier will check convergence status in the background
   *                          before the solver will move on to the next eigen-vector.
   */
  public HebbianSolver(int numPassesPerEigen) {
    this(0.0, numPassesPerEigen);
  }

  /**
   * Primary singular vector solving method.
   *
   * @param corpus input matrix to find singular vectors of.  Needs not be symmetric, should probably be sparse (in
   *   fact the input vectors are not mutated, and accessed only via dot-products and sums, so they should be
   *   {@link org.apache.mahout.math.SequentialAccessSparseVector }
   * @param desiredRank the number of singular vectors to find (in roughly decreasing order by singular value)
   * @return the final {@link TrainingState } of the solver, after desiredRank singular vectors (and approximate
   *         singular values) have been found.
   */
  public TrainingState solve(Matrix corpus,
                             int desiredRank) {
    int cols = corpus.numCols();
    Matrix eigens = new DenseMatrix(desiredRank, cols);
    List<Double> eigenValues = Lists.newArrayList();
    log.info("Finding {} singular vectors of matrix with {} rows, via Hebbian", desiredRank, corpus.numRows());
    /*
     * The corpusProjections matrix is a running cache of the residual projection of each corpus vector against all
     * of the previously found singular vectors.  Without this, if multiple passes over the data is made (per
     * singular vector), recalculating these projections eventually dominates the computational complexity of the
     * solver.
     */
    Matrix corpusProjections = new DenseMatrix(corpus.numRows(), desiredRank);
    TrainingState state = new TrainingState(eigens, corpusProjections);
    for (int i = 0; i < desiredRank; i++) {
      Vector currentEigen = new DenseVector(cols);
      Vector previousEigen = null;
      while (hasNotConverged(currentEigen, corpus, state)) {
        int randomStartingIndex = getRandomStartingIndex(corpus, eigens);
        Vector initialTrainingVector = corpus.viewRow(randomStartingIndex);
        state.setTrainingIndex(randomStartingIndex);
        updater.update(currentEigen, initialTrainingVector, state);
        for (int corpusRow = 0; corpusRow < corpus.numRows(); corpusRow++) {
          state.setTrainingIndex(corpusRow);
          if (corpusRow != randomStartingIndex) {
            updater.update(currentEigen, corpus.viewRow(corpusRow), state);
          }
        }
        state.setFirstPass(false);
        if (DEBUG) {
          if (previousEigen == null) {
            previousEigen = currentEigen.clone();
          } else {
            double dot = currentEigen.dot(previousEigen);
            if (dot > 0.0) {
              dot /= currentEigen.norm(2) * previousEigen.norm(2);
            }
           // log.info("Current pass * previous pass = {}", dot);
          }
        }
      }
      // converged!
      double eigenValue = state.getStatusProgress().get(state.getStatusProgress().size() - 1).getEigenValue();
      // it's actually more efficient to do this to normalize than to call currentEigen = currentEigen.normalize(),
      // because the latter does a clone, which isn't necessary here.
      currentEigen.assign(new TimesFunction(), 1 / currentEigen.norm(2));
      eigens.assignRow(i, currentEigen);
      eigenValues.add(eigenValue);
      state.setCurrentEigenValues(eigenValues);
      log.info("Found eigenvector {}, eigenvalue: {}", i, eigenValue);

      /**
       *  TODO: Persist intermediate output!
       */
      state.setFirstPass(true);
      state.setNumEigensProcessed(state.getNumEigensProcessed() + 1);
      state.setActivationDenominatorSquared(0);
      state.setActivationNumerator(0);
      state.getStatusProgress().clear();
      numPasses = 0;
    }
    return state;
  }

  /**
   * You have to start somewhere...
   * TODO: start instead wherever you find a vector with maximum residual length after subtracting off the projection
   * TODO: onto all previous eigenvectors.
   *
   * @param corpus the corpus matrix
   * @param eigens not currently used, but should be (see above TODO)
   * @return the index into the corpus where the "starting seed" input vector lies.
   */
  private int getRandomStartingIndex(Matrix corpus, Matrix eigens) {
    int index;
    Vector v;
    do {
      double r = rng.nextDouble();
      index = (int) (r * corpus.numRows());
      v = corpus.viewRow(index);
    } while (v == null || v.norm(2) == 0 || v.getNumNondefaultElements() < 5);
    return index;
  }

  /**
   * Uses the {@link SingularVectorVerifier } to check for convergence
   *
   * @param currentPseudoEigen the purported singular vector whose convergence is being checked
   * @param corpus             the corpus to check against
   * @param state              contains the previous eigens, various other solving state {@link TrainingState}
   * @return true if <em>either</em> we have converged, <em>or</em> maxPassesPerEigen has been exceeded.
   */
  protected boolean hasNotConverged(Vector currentPseudoEigen,
                                    Matrix corpus,
                                    TrainingState state) {
    numPasses++;
    if (state.isFirstPass()) {
      log.info("First pass through the corpus, no need to check convergence...");
      return true;
    }
    Matrix previousEigens = state.getCurrentEigens();
    log.info("Have made {} passes through the corpus, checking convergence...", numPasses);
    /*
     * Step 1: orthogonalize currentPseudoEigen by subtracting off eigen(i) * helper.get(i)
     * Step 2: zero-out the helper vector because it has already helped.
     */
    for (int i = 0; i < state.getNumEigensProcessed(); i++) {
      Vector previousEigen = previousEigens.viewRow(i);
      currentPseudoEigen.assign(previousEigen, new PlusMult(-state.getHelperVector().get(i)));
      state.getHelperVector().set(i, 0);
    }
    if (DEBUG && currentPseudoEigen.norm(2) > 0) {
      for (int i = 0; i < state.getNumEigensProcessed(); i++) {
        Vector previousEigen = previousEigens.viewRow(i);
        log.info("dot with previous: {}", previousEigen.dot(currentPseudoEigen) / currentPseudoEigen.norm(2));
      }
    }
    /*
     * Step 3: verify how eigen-like the prospective eigen is.  This is potentially asynchronous.
     */
    EigenStatus status = verify(corpus, currentPseudoEigen);
    if (status.inProgress()) {
      log.info("Verifier not finished, making another pass...");
    } else {
      log.info("Has 1 - cosAngle: {}, convergence target is: {}", 1.0 - status.getCosAngle(), convergenceTarget);
      state.getStatusProgress().add(status);
    }
    return
        state.getStatusProgress().size() <= maxPassesPerEigen
        && 1.0 - status.getCosAngle() > convergenceTarget;
  }

  protected EigenStatus verify(Matrix corpus, Vector currentPseudoEigen) {
    return verifier.verify(corpus, currentPseudoEigen);
  }

  public static void main(String[] args) {
    Properties props = new Properties();
    String propertiesFile = args.length > 0 ? args[0] : "config/solver.properties";
    //  props.load(new FileInputStream(propertiesFile));

    String corpusDir = props.getProperty("solver.input.dir");
    String outputDir = props.getProperty("solver.output.dir");
    if (corpusDir == null || corpusDir.isEmpty() || outputDir == null || outputDir.isEmpty()) {
      log.error("{} must contain values for solver.input.dir and solver.output.dir", propertiesFile);
      return;
    }
    //int inBufferSize = Integer.parseInt(props.getProperty("solver.input.bufferSize"));
    int rank = Integer.parseInt(props.getProperty("solver.output.desiredRank"));
    double convergence = Double.parseDouble(props.getProperty("solver.convergence"));
    int maxPasses = Integer.parseInt(props.getProperty("solver.maxPasses"));
    //int numThreads = Integer.parseInt(props.getProperty("solver.verifier.numThreads"));

    HebbianUpdater updater = new HebbianUpdater();
    SingularVectorVerifier verifier = new AsyncEigenVerifier();
    HebbianSolver solver = new HebbianSolver(updater, verifier, convergence, maxPasses);
    Matrix corpus = null;
    /*
    if (numThreads <= 1) {
      //  corpus = new DiskBufferedDoubleMatrix(new File(corpusDir), inBufferSize);
    } else {
      //  corpus = new ParallelMultiplyingDiskBufferedDoubleMatrix(new File(corpusDir), inBufferSize, numThreads);
    }
     */
    long now = System.currentTimeMillis();
    TrainingState finalState = solver.solve(corpus, rank);
    long time = (System.currentTimeMillis() - now) / 1000;
    log.info("Solved {} eigenVectors in {} seconds.  Persisted to {}",
             finalState.getCurrentEigens().rowSize(), time, outputDir);
  }

  
}
