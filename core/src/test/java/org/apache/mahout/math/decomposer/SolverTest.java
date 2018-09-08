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

package org.apache.mahout.math.decomposer;

import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Random;

public abstract class SolverTest extends MahoutTestCase {
  private static final Logger log = LoggerFactory.getLogger(SolverTest.class);

  public static void assertOrthonormal(Matrix eigens) {
    assertOrthonormal(eigens, 1.0e-6);
  }

  public static void assertOrthonormal(Matrix currentEigens, double errorMargin) {
    List<String> nonOrthogonals = Lists.newArrayList();
    for (int i = 0; i < currentEigens.numRows(); i++) {
      Vector ei = currentEigens.viewRow(i);
      for (int j = 0; j <= i; j++) {
        Vector ej = currentEigens.viewRow(j);
        if (ei.norm(2) == 0 || ej.norm(2) == 0) {
          continue;
        }
        double dot = ei.dot(ej);
        if (i == j) {
          assertTrue("not norm 1 : " + dot + " (eigen #" + i + ')', Math.abs(1.0 - dot) < errorMargin);
        } else {
          if (Math.abs(dot) > errorMargin) {
            log.info("not orthogonal : {} (eigens {}, {})", dot, i, j);
            nonOrthogonals.add("(" + i + ',' + j + ')');
          }
        }
      }
      log.info("{}:{}", nonOrthogonals.size(), nonOrthogonals);
    }
  }

  public static void assertOrthonormal(LanczosState state) {
    double errorMargin = 1.0e-5;
    List<String> nonOrthogonals = Lists.newArrayList();
    for (int i = 0; i < state.getIterationNumber(); i++) {
      Vector ei = state.getRightSingularVector(i);
      for (int j = 0; j <= i; j++) {
        Vector ej = state.getRightSingularVector(j);
        if (ei.norm(2) == 0 || ej.norm(2) == 0) {
          continue;
        }
        double dot = ei.dot(ej);
        if (i == j) {
          assertTrue("not norm 1 : " + dot + " (eigen #" + i + ')', Math.abs(1.0 - dot) < errorMargin);
        } else {
          if (Math.abs(dot) > errorMargin) {
            log.info("not orthogonal : {} (eigens {}, {})", dot, i, j);
            nonOrthogonals.add("(" + i + ',' + j + ')');
          }
        }
      }
      if (!nonOrthogonals.isEmpty()) {
        log.info("{}:{}", nonOrthogonals.size(), nonOrthogonals);
      }
    }
  }

  public static void assertEigen(Matrix eigens, VectorIterable corpus, double errorMargin, boolean isSymmetric) {
    assertEigen(eigens, corpus, eigens.numRows(), errorMargin, isSymmetric);
  }

  public static void assertEigen(Matrix eigens,
                                 VectorIterable corpus,
                                 int numEigensToCheck,
                                 double errorMargin,
                                 boolean isSymmetric) {
    for (int i = 0; i < numEigensToCheck; i++) {
      Vector e = eigens.viewRow(i);
      assertEigen(i, e, corpus, errorMargin, isSymmetric);
    }
  }

  public static void assertEigen(int i, Vector e, VectorIterable corpus, double errorMargin,
      boolean isSymmetric) {
    if (e.getLengthSquared() == 0) {
      return;
    }
    Vector afterMultiply = isSymmetric ? corpus.times(e) : corpus.timesSquared(e);
    double dot = afterMultiply.dot(e);
    double afterNorm = afterMultiply.getLengthSquared();
    double error = 1 - Math.abs(dot / Math.sqrt(afterNorm * e.getLengthSquared()));
    log.info("the eigen-error: {} for eigen {}", error, i);
    assertTrue("Error: {" + error + " too high! (for eigen " + i + ')', Math.abs(error) < errorMargin);
  }

  /**
   * Builds up a consistently random (same seed every time) sparse matrix, with sometimes
   * repeated rows.
   */
  public static Matrix randomSequentialAccessSparseMatrix(int numRows,
                                                          int nonNullRows,
                                                          int numCols,
                                                          int entriesPerRow,
                                                          double entryMean) {
    Matrix m = new SparseRowMatrix(numRows, numCols);
    //double n = 0;
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < nonNullRows; i++) {
      Vector v = new SequentialAccessSparseVector(numCols);
      for (int j = 0; j < entriesPerRow; j++) {
        int col = r.nextInt(numCols);
        double val = r.nextGaussian();
        v.set(col, val * entryMean);
      }
      int c = r.nextInt(numRows);
      if (r.nextBoolean() || numRows == nonNullRows) {
        m.assignRow(numRows == nonNullRows ? i : c, v);
      } else {
        Vector other = m.viewRow(r.nextInt(numRows));
        if (other != null && other.getLengthSquared() > 0) {
          m.assignRow(c, other.clone());
        }
      }
      //n += m.getRow(c).getLengthSquared();
    }
    return m;
  }

  public static Matrix randomHierarchicalMatrix(int numRows, int numCols, boolean symmetric) {
    Matrix matrix = new DenseMatrix(numRows, numCols);
    // TODO rejigger tests so that it doesn't expect this particular seed
    Random r = new Random(1234L);
    for (int row = 0; row < numRows; row++) {
      Vector v = new DenseVector(numCols);
      for (int col = 0; col < numCols; col++) {
        double val = r.nextGaussian();
        v.set(col, val);
      }
      v.assign(Functions.MULT, 1/((row + 1) * v.norm(2)));
      matrix.assignRow(row, v);
    }
    if (symmetric) {
      return matrix.times(matrix.transpose());
    }
    return matrix;
  }

  public static Matrix randomHierarchicalSymmetricMatrix(int size) {
    return randomHierarchicalMatrix(size, size, true);
  }
}
