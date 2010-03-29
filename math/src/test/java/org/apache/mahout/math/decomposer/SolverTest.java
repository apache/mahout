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

import junit.framework.TestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;


public abstract class SolverTest extends TestCase {

  private static final Logger log = LoggerFactory.getLogger(SolverTest.class);

  protected SolverTest(String name) {
    super(name);
  }

  public static void assertOrthonormal(Matrix eigens) {
    assertOrthonormal(eigens, 1.0e-6);
  }

  public static void assertOrthonormal(Matrix currentEigens, double errorMargin) {
    for (int i = 0; i < currentEigens.numRows(); i++) {
      Vector ei = currentEigens.getRow(i);
      for (int j = 0; j <= i; j++) {
        Vector ej = currentEigens.getRow(j);
        if (ei.norm(2) == 0 || ej.norm(2) == 0) {
          continue;
        }
        double dot = ei.dot(ej);
        if (i == j) {
          assertTrue("not norm 1 : " + dot + " (eigen #" + i + ')', (Math.abs(1 - dot) < errorMargin));
        } else {
          assertTrue("not orthogonal : " + dot + " (eigens " + i + ", " + j + ')', Math.abs(dot) < errorMargin);
        }
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
      Vector e = eigens.getRow(i);
      if (e.getLengthSquared() == 0) continue;
      Vector afterMultiply = isSymmetric ? corpus.times(e) : corpus.timesSquared(e);
      double dot = afterMultiply.dot(e);
      double afterNorm = afterMultiply.getLengthSquared();
      double error = 1 - dot / Math.sqrt(afterNorm * e.getLengthSquared());
      log.info("Eigenvalue({}) = {}", i, Math.sqrt(afterNorm/e.getLengthSquared()));
      assertTrue("Error margin: {" + error + " too high! (for eigen " + i + ')', Math.abs(error) < errorMargin);
    }
  }

  /**
   * Builds up a consistently random (same seed every time) sparse matrix, with sometimes
   * repeated rows.
   * @param numRows
   * @param nonNullRows
   * @param numCols
   * @param entriesPerRow
   * @param entryMean
   * @return
   */
  public static Matrix randomSequentialAccessSparseMatrix(int numRows,
                                                          int nonNullRows,
                                                          int numCols,
                                                          int entriesPerRow,
                                                          double entryMean) {
    SparseRowMatrix m = new SparseRowMatrix(new int[]{numRows, numCols});
    double n = 0;
    Random r = new Random(1234L);
    for (int i = 0; i < nonNullRows; i++) {
      SequentialAccessSparseVector v = new SequentialAccessSparseVector(numCols);
      for (int j = 0; j < entriesPerRow; j++) {
        int col = r.nextInt(numCols);
        double val = r.nextGaussian();
        v.set(col, val * entryMean);
      }
      int c = r.nextInt(numRows);
      if (r.nextBoolean() || numRows == nonNullRows) {
        m.assignRow(numRows == nonNullRows ? i : c, v);
      } else {
        Vector other = m.getRow(r.nextInt(numRows));
        if (other != null && other.getLengthSquared() > 0) {
          m.assignRow(c, other.clone());
        }
      }
      n += m.getRow(c).getLengthSquared();
    }
    return m;
  }
}
