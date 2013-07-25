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

package org.apache.mahout.math.decomposer.lanczos;


import java.util.EnumMap;
import java.util.Map;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.solver.EigenDecomposition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Simple implementation of the <a href="http://en.wikipedia.org/wiki/Lanczos_algorithm">Lanczos algorithm</a> for
 * finding eigenvalues of a symmetric matrix, applied to non-symmetric matrices by applying Matrix.timesSquared(vector)
 * as the "matrix-multiplication" method.</p>
 *
 * See the SSVD code for a better option
 * {@link org.apache.mahout.math.ssvd.SequentialBigSvd}
 * See also the docs on
 * <a href=https://cwiki.apache.org/confluence/display/MAHOUT/Stochastic+Singular+Value+Decomposition>stochastic
 * projection SVD</a>
 * <p>
 * To avoid floating point overflow problems which arise in power-methods like Lanczos, an initial pass is made
 * through the input matrix to
 * <ul>
 *   <li>generate a good starting seed vector by summing all the rows of the input matrix, and</li>
 *   <li>compute the trace(inputMatrix<sup>t</sup>*matrix)
 * </ul>
 * </p>
 * <p>
 * This latter value, being the sum of all of the singular values, is used to rescale the entire matrix, effectively
 * forcing the largest singular value to be strictly less than one, and transforming floating point <em>overflow</em>
 * problems into floating point <em>underflow</em> (ie, very small singular values will become invisible, as they
 * will appear to be zero and the algorithm will terminate).
 * </p>
 * <p>This implementation uses {@link EigenDecomposition} to do the
 * eigenvalue extraction from the small (desiredRank x desiredRank) tridiagonal matrix.  Numerical stability is
 * achieved via brute-force: re-orthogonalization against all previous eigenvectors is computed after every pass.
 * This can be made smarter if (when!) this proves to be a major bottleneck.  Of course, this step can be parallelized
 * as well.
 * </p>
 * @see org.apache.mahout.math.ssvd.SequentialBigSvd
 */
public class LanczosSolver {

  private static final Logger log = LoggerFactory.getLogger(LanczosSolver.class);

  public static final double SAFE_MAX = 1.0e150;

  public enum TimingSection {
    ITERATE, ORTHOGANLIZE, TRIDIAG_DECOMP, FINAL_EIGEN_CREATE
  }

  private final Map<TimingSection, Long> startTimes = new EnumMap<TimingSection, Long>(TimingSection.class);
  private final Map<TimingSection, Long> times = new EnumMap<TimingSection, Long>(TimingSection.class);

  private static final class Scale extends DoubleFunction {
    private final double d;

    private Scale(double d) {
      this.d = d;
    }

    @Override
    public double apply(double arg1) {
      return arg1 * d;
    }
  }

  public void solve(LanczosState state,
                    int desiredRank) {
    solve(state, desiredRank, false);
  }

  public void solve(LanczosState state,
                    int desiredRank,
                    boolean isSymmetric) {
    VectorIterable corpus = state.getCorpus();
    log.info("Finding {} singular vectors of matrix with {} rows, via Lanczos",
        desiredRank, corpus.numRows());
    int i = state.getIterationNumber();
    Vector currentVector = state.getBasisVector(i - 1);
    Vector previousVector = state.getBasisVector(i - 2);
    double beta = 0;
    Matrix triDiag = state.getDiagonalMatrix();
    while (i < desiredRank) {
      startTime(TimingSection.ITERATE);
      Vector nextVector = isSymmetric ? corpus.times(currentVector) : corpus.timesSquared(currentVector);
      log.info("{} passes through the corpus so far...", i);
      if (state.getScaleFactor() <= 0) {
        state.setScaleFactor(calculateScaleFactor(nextVector));
      }
      nextVector.assign(new Scale(1.0 / state.getScaleFactor()));
      if (previousVector != null) {
        nextVector.assign(previousVector, new PlusMult(-beta));
      }
      // now orthogonalize
      double alpha = currentVector.dot(nextVector);
      nextVector.assign(currentVector, new PlusMult(-alpha));
      endTime(TimingSection.ITERATE);
      startTime(TimingSection.ORTHOGANLIZE);
      orthoganalizeAgainstAllButLast(nextVector, state);
      endTime(TimingSection.ORTHOGANLIZE);
      // and normalize
      beta = nextVector.norm(2);
      if (outOfRange(beta) || outOfRange(alpha)) {
        log.warn("Lanczos parameters out of range: alpha = {}, beta = {}.  Bailing out early!",
            alpha, beta);
        break;
      }
      nextVector.assign(new Scale(1 / beta));
      state.setBasisVector(i, nextVector);
      previousVector = currentVector;
      currentVector = nextVector;
      // save the projections and norms!
      triDiag.set(i - 1, i - 1, alpha);
      if (i < desiredRank - 1) {
        triDiag.set(i - 1, i, beta);
        triDiag.set(i, i - 1, beta);
      }
      state.setIterationNumber(++i);
    }
    startTime(TimingSection.TRIDIAG_DECOMP);

    log.info("Lanczos iteration complete - now to diagonalize the tri-diagonal auxiliary matrix.");
    // at this point, have tridiag all filled out, and basis is all filled out, and orthonormalized
    EigenDecomposition decomp = new EigenDecomposition(triDiag);

    Matrix eigenVects = decomp.getV();
    Vector eigenVals = decomp.getRealEigenvalues();
    endTime(TimingSection.TRIDIAG_DECOMP);
    startTime(TimingSection.FINAL_EIGEN_CREATE);
    for (int row = 0; row < i; row++) {
      Vector realEigen = null;

      Vector ejCol = eigenVects.viewColumn(row);
      int size = Math.min(ejCol.size(), state.getBasisSize());
      for (int j = 0; j < size; j++) {
        double d = ejCol.get(j);
        Vector rowJ = state.getBasisVector(j);
        if (realEigen == null) {
          realEigen = rowJ.like();
        }
        realEigen.assign(rowJ, new PlusMult(d));
      }

      Preconditions.checkState(realEigen != null);
      assert realEigen != null;

      realEigen = realEigen.normalize();
      state.setRightSingularVector(row, realEigen);
      double e = eigenVals.get(row) * state.getScaleFactor();
      if (!isSymmetric) {
        e = Math.sqrt(e);
      }
      log.info("Eigenvector {} found with eigenvalue {}", row, e);
      state.setSingularValue(row, e);
    }
    log.info("LanczosSolver finished.");
    endTime(TimingSection.FINAL_EIGEN_CREATE);
  }

  protected static double calculateScaleFactor(Vector nextVector) {
    return nextVector.norm(2);
  }

  private static boolean outOfRange(double d) {
    return Double.isNaN(d) || d > SAFE_MAX || -d > SAFE_MAX;
  }

  protected static void orthoganalizeAgainstAllButLast(Vector nextVector, LanczosState state) {
    for (int i = 0; i < state.getIterationNumber(); i++) {
      Vector basisVector = state.getBasisVector(i);
      double alpha;
      if (basisVector == null || (alpha = nextVector.dot(basisVector)) == 0.0) {
        continue;
      }
      nextVector.assign(basisVector, new PlusMult(-alpha));
    }
  }

  private void startTime(TimingSection section) {
    startTimes.put(section, System.nanoTime());
  }

  private void endTime(TimingSection section) {
    if (!times.containsKey(section)) {
      times.put(section, 0L);
    }
    times.put(section, times.get(section) + System.nanoTime() - startTimes.get(section));
  }

}
