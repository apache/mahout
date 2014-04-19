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

package org.apache.mahout.math.hadoop.stochasticsvd.qr;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.OrderedIntDoubleMapping;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.UpperTriangular;

/**
 * Givens Thin solver. Standard Givens operations are reordered in a way that
 * helps us to push them thru MapReduce operations in a block fashion.
 */
public class GivensThinSolver {

  private double[] vARow;
  private double[] vQtRow;
  private final double[][] mQt;
  private final double[][] mR;
  private int qtStartRow;
  private int rStartRow;
  private int m;
  private final int n; // m-row cnt, n- column count, m>=n
  private int cnt;
  private final double[] cs = new double[2];

  public GivensThinSolver(int m, int n) {
    if (!(m >= n)) {
      throw new IllegalArgumentException("Givens thin QR: must be true: m>=n");
    }

    this.m = m;
    this.n = n;

    mQt = new double[n][];
    mR = new double[n][];
    vARow = new double[n];
    vQtRow = new double[m];

    for (int i = 0; i < n; i++) {
      mQt[i] = new double[this.m];
      mR[i] = new double[this.n];
    }
    cnt = 0;
  }

  public void reset() {
    cnt = 0;
  }

  public void solve(Matrix a) {

    assert a.rowSize() == m;
    assert a.columnSize() == n;

    double[] aRow = new double[n];
    for (int i = 0; i < m; i++) {
      Vector aRowV = a.viewRow(i);
      for (int j = 0; j < n; j++) {
        aRow[j] = aRowV.getQuick(j);
      }
      appendRow(aRow);
    }
  }

  public boolean isFull() {
    return cnt == m;
  }

  public int getM() {
    return m;
  }

  public int getN() {
    return n;
  }

  public int getCnt() {
    return cnt;
  }

  public void adjust(int newM) {
    if (newM == m) {
      // no adjustment is required.
      return; 
    }
    if (newM < n) {
      throw new IllegalArgumentException("new m can't be less than n");
    }
    if (newM < cnt) {
      throw new IllegalArgumentException(
          "new m can't be less than rows accumulated");
    }
    vQtRow = new double[newM];

    // grow or shrink qt rows
    if (newM > m) {
      // grow qt rows
      for (int i = 0; i < n; i++) {
        mQt[i] = Arrays.copyOf(mQt[i], newM);
        System.arraycopy(mQt[i], 0, mQt[i], newM - m, m);
        Arrays.fill(mQt[i], 0, newM - m, 0);
      }
    } else {
      // shrink qt rows
      for (int i = 0; i < n; i++) {
        mQt[i] = Arrays.copyOfRange(mQt[i], m - newM, m);
      }
    }

    m = newM;

  }

  public void trim() {
    adjust(cnt);
  }

  /**
   * api for row-by-row addition
   * 
   * @param aRow
   */
  public void appendRow(double[] aRow) {
    if (cnt >= m) {
      throw new IllegalStateException("thin QR solver fed more rows than initialized for");
    }
    try {
      /*
       * moving pointers around is inefficient but for the sanity's sake i am
       * keeping it this way so i don't have to guess how R-tilde index maps to
       * actual block index
       */
      Arrays.fill(vQtRow, 0);
      vQtRow[m - cnt - 1] = 1;
      int height = cnt > n ? n : cnt;
      System.arraycopy(aRow, 0, vARow, 0, n);

      if (height > 0) {
        givens(vARow[0], getRRow(0)[0], cs);
        applyGivensInPlace(cs[0], cs[1], vARow, getRRow(0), 0, n);
        applyGivensInPlace(cs[0], cs[1], vQtRow, getQtRow(0), 0, m);
      }

      for (int i = 1; i < height; i++) {
        givens(getRRow(i - 1)[i], getRRow(i)[i], cs);
        applyGivensInPlace(cs[0], cs[1], getRRow(i - 1), getRRow(i), i,
            n - i);
        applyGivensInPlace(cs[0], cs[1], getQtRow(i - 1), getQtRow(i), 0,
            m);
      }
      /*
       * push qt and r-tilde 1 row down
       * 
       * just swap the references to reduce GC churning
       */
      pushQtDown();
      double[] swap = getQtRow(0);
      setQtRow(0, vQtRow);
      vQtRow = swap;

      pushRDown();
      swap = getRRow(0);
      setRRow(0, vARow);
      vARow = swap;

    } finally {
      cnt++;
    }
  }

  private double[] getQtRow(int row) {

    return mQt[(row += qtStartRow) >= n ? row - n : row];
  }

  private void setQtRow(int row, double[] qtRow) {
    mQt[(row += qtStartRow) >= n ? row - n : row] = qtRow;
  }

  private void pushQtDown() {
    qtStartRow = qtStartRow == 0 ? n - 1 : qtStartRow - 1;
  }

  private double[] getRRow(int row) {
    row += rStartRow;
    return mR[row >= n ? row - n : row];
  }

  private void setRRow(int row, double[] rrow) {
    mR[(row += rStartRow) >= n ? row - n : row] = rrow;
  }

  private void pushRDown() {
    rStartRow = rStartRow == 0 ? n - 1 : rStartRow - 1;
  }

  /*
   * warning: both of these return actually n+1 rows with the last one being //
   * not interesting.
   */
  public UpperTriangular getRTilde() {
    UpperTriangular packedR = new UpperTriangular(n);
    for (int i = 0; i < n; i++) {
      packedR.assignNonZeroElementsInRow(i, getRRow(i));
    }
    return packedR;
  }

  public double[][] getThinQtTilde() {
    if (qtStartRow != 0) {
      /*
       * rotate qt rows into place
       * 
       * double[~500][], once per block, not a big deal.
       */
      double[][] qt = new double[n][]; 
      System.arraycopy(mQt, qtStartRow, qt, 0, n - qtStartRow);
      System.arraycopy(mQt, 0, qt, n - qtStartRow, qtStartRow);
      return qt;
    }
    return mQt;
  }

  public static void applyGivensInPlace(double c, double s, double[] row1,
      double[] row2, int offset, int len) {

    int n = offset + len;
    for (int j = offset; j < n; j++) {
      double tau1 = row1[j];
      double tau2 = row2[j];
      row1[j] = c * tau1 - s * tau2;
      row2[j] = s * tau1 + c * tau2;
    }
  }

  public static void applyGivensInPlace(double c, double s, Vector row1,
      Vector row2, int offset, int len) {

    int n = offset + len;
    for (int j = offset; j < n; j++) {
      double tau1 = row1.getQuick(j);
      double tau2 = row2.getQuick(j);
      row1.setQuick(j, c * tau1 - s * tau2);
      row2.setQuick(j, s * tau1 + c * tau2);
    }
  }

  public static void applyGivensInPlace(double c, double s, int i, int k,
      Matrix mx) {
    int n = mx.columnSize();

    for (int j = 0; j < n; j++) {
      double tau1 = mx.get(i, j);
      double tau2 = mx.get(k, j);
      mx.set(i, j, c * tau1 - s * tau2);
      mx.set(k, j, s * tau1 + c * tau2);
    }
  }

  public static void fromRho(double rho, double[] csOut) {
    if (rho == 1) {
      csOut[0] = 0;
      csOut[1] = 1;
      return;
    }
    if (Math.abs(rho) < 1) {
      csOut[1] = 2 * rho;
      csOut[0] = Math.sqrt(1 - csOut[1] * csOut[1]);
      return;
    }
    csOut[0] = 2 / rho;
    csOut[1] = Math.sqrt(1 - csOut[0] * csOut[0]);
  }

  public static void givens(double a, double b, double[] csOut) {
    if (b == 0) {
      csOut[0] = 1;
      csOut[1] = 0;
      return;
    }
    if (Math.abs(b) > Math.abs(a)) {
      double tau = -a / b;
      csOut[1] = 1 / Math.sqrt(1 + tau * tau);
      csOut[0] = csOut[1] * tau;
    } else {
      double tau = -b / a;
      csOut[0] = 1 / Math.sqrt(1 + tau * tau);
      csOut[1] = csOut[0] * tau;
    }
  }

  public static double toRho(double c, double s) {
    if (c == 0) {
      return 1;
    }
    if (Math.abs(s) < Math.abs(c)) {
      return Math.signum(c) * s / 2;
    } else {
      return Math.signum(s) * 2 / c;
    }
  }

  public static void mergeR(UpperTriangular r1, UpperTriangular r2) {
    TriangularRowView r1Row = new TriangularRowView(r1);
    TriangularRowView r2Row = new TriangularRowView(r2);
    
    int kp = r1Row.size();
    assert kp == r2Row.size();

    double[] cs = new double[2];

    for (int v = 0; v < kp; v++) {
      for (int u = v; u < kp; u++) {
        givens(r1Row.setViewedRow(u).get(u), r2Row.setViewedRow(u - v).get(u),
            cs);
        applyGivensInPlace(cs[0], cs[1], r1Row, r2Row, u, kp - u);
      }
    }
  }

  public static void mergeR(double[][] r1, double[][] r2) {
    int kp = r1[0].length;
    assert kp == r2[0].length;

    double[] cs = new double[2];

    for (int v = 0; v < kp; v++) {
      for (int u = v; u < kp; u++) {
        givens(r1[u][u], r2[u - v][u], cs);
        applyGivensInPlace(cs[0], cs[1], r1[u], r2[u - v], u, kp - u);
      }
    }

  }

  public static void mergeRonQ(UpperTriangular r1, UpperTriangular r2,
      double[][] qt1, double[][] qt2) {
    TriangularRowView r1Row = new TriangularRowView(r1);
    TriangularRowView r2Row = new TriangularRowView(r2);
    int kp = r1Row.size();
    assert kp == r2Row.size();
    assert kp == qt1.length;
    assert kp == qt2.length;

    int r = qt1[0].length;
    assert qt2[0].length == r;

    double[] cs = new double[2];

    for (int v = 0; v < kp; v++) {
      for (int u = v; u < kp; u++) {
        givens(r1Row.setViewedRow(u).get(u), r2Row.setViewedRow(u - v).get(u),
            cs);
        applyGivensInPlace(cs[0], cs[1], r1Row, r2Row, u, kp - u);
        applyGivensInPlace(cs[0], cs[1], qt1[u], qt2[u - v], 0, r);
      }
    }
  }

  public static void mergeRonQ(double[][] r1, double[][] r2, double[][] qt1,
      double[][] qt2) {

    int kp = r1[0].length;
    assert kp == r2[0].length;
    assert kp == qt1.length;
    assert kp == qt2.length;

    int r = qt1[0].length;
    assert qt2[0].length == r;
    double[] cs = new double[2];

    /*
     * pairwise givens(a,b) so that a come off main diagonal in r1 and bs come
     * off u-th upper subdiagonal in r2.
     */
    for (int v = 0; v < kp; v++) {
      for (int u = v; u < kp; u++) {
        givens(r1[u][u], r2[u - v][u], cs);
        applyGivensInPlace(cs[0], cs[1], r1[u], r2[u - v], u, kp - u);
        applyGivensInPlace(cs[0], cs[1], qt1[u], qt2[u - v], 0, r);
      }
    }
  }

  // returns merged Q (which in this case is the qt1)
  public static double[][] mergeQrUp(double[][] qt1, double[][] r1,
      double[][] r2) {
    int kp = qt1.length;
    int r = qt1[0].length;

    double[][] qTilde = new double[kp][];
    for (int i = 0; i < kp; i++) {
      qTilde[i] = new double[r];
    }
    mergeRonQ(r1, r2, qt1, qTilde);
    return qt1;
  }

  // returns merged Q (which in this case is the qt1)
  public static double[][] mergeQrUp(double[][] qt1, UpperTriangular r1, UpperTriangular r2) {
    int kp = qt1.length;
    int r = qt1[0].length;

    double[][] qTilde = new double[kp][];
    for (int i = 0; i < kp; i++) {
      qTilde[i] = new double[r];
    }
    mergeRonQ(r1, r2, qt1, qTilde);
    return qt1;
  }

  public static double[][] mergeQrDown(double[][] r1, double[][] qt2, double[][] r2) {
    int kp = qt2.length;
    int r = qt2[0].length;

    double[][] qTilde = new double[kp][];
    for (int i = 0; i < kp; i++) {
      qTilde[i] = new double[r];
    }
    mergeRonQ(r1, r2, qTilde, qt2);
    return qTilde;

  }

  public static double[][] mergeQrDown(UpperTriangular r1, double[][] qt2, UpperTriangular r2) {
    int kp = qt2.length;
    int r = qt2[0].length;

    double[][] qTilde = new double[kp][];
    for (int i = 0; i < kp; i++) {
      qTilde[i] = new double[r];
    }
    mergeRonQ(r1, r2, qTilde, qt2);
    return qTilde;

  }

  public static double[][] computeQtHat(double[][] qt, int i,
      Iterator<UpperTriangular> rIter) {
    UpperTriangular rTilde = rIter.next();
    for (int j = 1; j < i; j++) {
      mergeR(rTilde, rIter.next());
    }
    if (i > 0) {
      qt = mergeQrDown(rTilde, qt, rIter.next());
    }
    while (rIter.hasNext()) {
      qt = mergeQrUp(qt, rTilde, rIter.next());
    }
    return qt;
  }

  // test helpers
  public static boolean isOrthonormal(double[][] qt, boolean insufficientRank, double epsilon) {
    int n = qt.length;
    int rank = 0;
    for (int i = 0; i < n; i++) {
      Vector ei = new DenseVector(qt[i], true);

      double norm = ei.norm(2);

      if (Math.abs(1.0 - norm) < epsilon) {
        rank++;
      } else if (Math.abs(norm) > epsilon) {
        return false; // not a rank deficiency, either
      }

      for (int j = 0; j <= i; j++) {
        Vector ej = new DenseVector(qt[j], true);
        double dot = ei.dot(ej);
        if (!(Math.abs((i == j && rank > j ? 1.0 : 0.0) - dot) < epsilon)) {
          return false;
        }
      }
    }
    return insufficientRank ? rank < n : rank == n;
  }

  public static boolean isOrthonormalBlocked(Iterable<double[][]> qtHats,
      boolean insufficientRank, double epsilon) {
    int n = qtHats.iterator().next().length;
    int rank = 0;
    for (int i = 0; i < n; i++) {
      List<Vector> ei = Lists.newArrayList();
      // Vector e_i=new DenseVector (qt[i],true);
      for (double[][] qtHat : qtHats) {
        ei.add(new DenseVector(qtHat[i], true));
      }

      double norm = 0;
      for (Vector v : ei) {
        norm += v.dot(v);
      }
      norm = Math.sqrt(norm);
      if (Math.abs(1 - norm) < epsilon) {
        rank++;
      } else if (Math.abs(norm) > epsilon) {
        return false; // not a rank deficiency, either
      }

      for (int j = 0; j <= i; j++) {
        List<Vector> ej = Lists.newArrayList();
        for (double[][] qtHat : qtHats) {
          ej.add(new DenseVector(qtHat[j], true));
        }

        // Vector e_j = new DenseVector ( qt[j], true);
        double dot = 0;
        for (int k = 0; k < ei.size(); k++) {
          dot += ei.get(k).dot(ej.get(k));
        }
        if (!(Math.abs((i == j && rank > j ? 1 : 0) - dot) < epsilon)) {
          return false;
        }
      }
    }
    return insufficientRank ? rank < n : rank == n;
  }

  private static final class TriangularRowView extends AbstractVector {
    private final UpperTriangular viewed;
    private int rowNum;

    private TriangularRowView(UpperTriangular viewed) {
      super(viewed.columnSize());
      this.viewed = viewed;

    }

    TriangularRowView setViewedRow(int row) {
      rowNum = row;
      return this;
    }

    @Override
    public boolean isDense() {
      return true;
    }

    @Override
    public boolean isSequentialAccess() {
      return false;
    }

    @Override
    public Iterator<Element> iterator() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<Element> iterateNonZero() {
      throw new UnsupportedOperationException();
    }

    @Override
    public double getQuick(int index) {
      return viewed.getQuick(rowNum, index);
    }

    @Override
    public Vector like() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void setQuick(int index, double value) {
      viewed.setQuick(rowNum, index, value);

    }

    @Override
    public int getNumNondefaultElements() {
      throw new UnsupportedOperationException();
    }

    @Override
    public double getLookupCost() {
      return 1;
    }

    @Override
    public double getIteratorAdvanceCost() {
      return 1;
    }

    @Override
    public boolean isAddConstantTime() {
      return true;
    }

    @Override
    public Matrix matrixLike(int rows, int columns) {
      throw new UnsupportedOperationException();
    }

    /**
     * Used internally by assign() to update multiple indices and values at once.
     * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
     * <p/>
     * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
     *
     * @param updates a mapping of indices to values to merge in the vector.
     */
    @Override
    public void mergeUpdates(OrderedIntDoubleMapping updates) {
      int[] indices = updates.getIndices();
      double[] values = updates.getValues();
      for (int i = 0; i < updates.getNumMappings(); ++i) {
        viewed.setQuick(rowNum, indices[i], values[i]);
      }
    }

  }

}
