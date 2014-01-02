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
 *
 * Copyright 1999 CERN - European Organization for Nuclear Research.
 * Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
 * is hereby granted without fee, provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear in supporting documentation.
 * CERN makes no representations about the suitability of this software for any purpose.
 * It is provided "as is" without expressed or implied warranty.
 */
package org.apache.mahout.math;

public class SingularValueDecomposition implements java.io.Serializable {
  
  /** Arrays for internal storage of U and V. */
  private final double[][] u;
  private final double[][] v;
  
  /** Array for internal storage of singular values. */
  private final double[] s;
  
  /** Row and column dimensions. */
  private final int m;
  private final int n;
  
  /**To handle the case where numRows() < numCols() and to use the fact that SVD(A')=VSU'=> SVD(A')'=SVD(A)**/
  private boolean transpositionNeeded;
  
  /**
   * Constructs and returns a new singular value decomposition object; The
   * decomposed matrices can be retrieved via instance methods of the returned
   * decomposition object.
   * 
   * @param arg
   *            A rectangular matrix.
   */
  public SingularValueDecomposition(Matrix arg) {
    if (arg.numRows() < arg.numCols()) {
      transpositionNeeded = true;
    }
    
    // Derived from LINPACK code.
    // Initialize.
    double[][] a;
    if (transpositionNeeded) {
      //use the transpose Matrix
      m = arg.numCols();
      n = arg.numRows();
      a = new double[m][n];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          a[i][j] = arg.get(j, i);
        }
      }
    } else {
      m = arg.numRows();
      n = arg.numCols();
      a = new double[m][n];
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          a[i][j] = arg.get(i, j);
        }
      }
    }
    
    
    int nu = Math.min(m, n);
    s = new double[Math.min(m + 1, n)];
    u = new double[m][nu];
    v = new double[n][n];
    double[] e = new double[n];
    double[] work = new double[m];
    boolean wantu = true;
    boolean wantv = true;
    
    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    
    int nct = Math.min(m - 1, n);
    int nrt = Math.max(0, Math.min(n - 2, m));
    for (int k = 0; k < Math.max(nct, nrt); k++) {
      if (k < nct) {
        
        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        s[k] = 0;
        for (int i = k; i < m; i++) {
          s[k] = Algebra.hypot(s[k], a[i][k]);
        }
        if (s[k] != 0.0) {
          if (a[k][k] < 0.0) {
            s[k] = -s[k];
          }
          for (int i = k; i < m; i++) {
            a[i][k] /= s[k];
          }
          a[k][k] += 1.0;
        }
        s[k] = -s[k];
      }
      for (int j = k + 1; j < n; j++) {
        if (k < nct && s[k] != 0.0) {
          
          // Apply the transformation.
          
          double t = 0;
          for (int i = k; i < m; i++) {
            t += a[i][k] * a[i][j];
          }
          t = -t / a[k][k];
          for (int i = k; i < m; i++) {
            a[i][j] += t * a[i][k];
          }
        }
        
        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.
        
        e[j] = a[k][j];
      }
      if (wantu && k < nct) {
        
        // Place the transformation in U for subsequent back
        // multiplication.
        
        for (int i = k; i < m; i++) {
          u[i][k] = a[i][k];
        }
      }
      if (k < nrt) {
        
        // Compute the k-th row transformation and place the
        // k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        e[k] = 0;
        for (int i = k + 1; i < n; i++) {
          e[k] = Algebra.hypot(e[k], e[i]);
        }
        if (e[k] != 0.0) {
          if (e[k + 1] < 0.0) {
            e[k] = -e[k];
          }
          for (int i = k + 1; i < n; i++) {
            e[i] /= e[k];
          }
          e[k + 1] += 1.0;
        }
        e[k] = -e[k];
        if (k + 1 < m && e[k] != 0.0) {
          
          // Apply the transformation.
          
          for (int i = k + 1; i < m; i++) {
            work[i] = 0.0;
          }
          for (int j = k + 1; j < n; j++) {
            for (int i = k + 1; i < m; i++) {
              work[i] += e[j] * a[i][j];
            }
          }
          for (int j = k + 1; j < n; j++) {
            double t = -e[j] / e[k + 1];
            for (int i = k + 1; i < m; i++) {
              a[i][j] += t * work[i];
            }
          }
        }
        if (wantv) {
          
          // Place the transformation in V for subsequent
          // back multiplication.
          
          for (int i = k + 1; i < n; i++) {
            v[i][k] = e[i];
          }
        }
      }
    }
    
    // Set up the final bidiagonal matrix or order p.
    
    int p = Math.min(n, m + 1);
    if (nct < n) {
      s[nct] = a[nct][nct];
    }
    if (m < p) {
      s[p - 1] = 0.0;
    }
    if (nrt + 1 < p) {
      e[nrt] = a[nrt][p - 1];
    }
    e[p - 1] = 0.0;
    
    // If required, generate U.
    
    if (wantu) {
      for (int j = nct; j < nu; j++) {
        for (int i = 0; i < m; i++) {
          u[i][j] = 0.0;
        }
        u[j][j] = 1.0;
      }
      for (int k = nct - 1; k >= 0; k--) {
        if (s[k] != 0.0) {
          for (int j = k + 1; j < nu; j++) {
            double t = 0;
            for (int i = k; i < m; i++) {
              t += u[i][k] * u[i][j];
            }
            t = -t / u[k][k];
            for (int i = k; i < m; i++) {
              u[i][j] += t * u[i][k];
            }
          }
          for (int i = k; i < m; i++) {
            u[i][k] = -u[i][k];
          }
          u[k][k] = 1.0 + u[k][k];
          for (int i = 0; i < k - 1; i++) {
            u[i][k] = 0.0;
          }
        } else {
          for (int i = 0; i < m; i++) {
            u[i][k] = 0.0;
          }
          u[k][k] = 1.0;
        }
      }
    }
    
    // If required, generate V.
    
    if (wantv) {
      for (int k = n - 1; k >= 0; k--) {
        if (k < nrt && e[k] != 0.0) {
          for (int j = k + 1; j < nu; j++) {
            double t = 0;
            for (int i = k + 1; i < n; i++) {
              t += v[i][k] * v[i][j];
            }
            t = -t / v[k + 1][k];
            for (int i = k + 1; i < n; i++) {
              v[i][j] += t * v[i][k];
            }
          }
        }
        for (int i = 0; i < n; i++) {
          v[i][k] = 0.0;
        }
        v[k][k] = 1.0;
      }
    }
    
    // Main iteration loop for the singular values.
    
    int pp = p - 1;
    int iter = 0;
    double eps = Math.pow(2.0, -52.0);
    double tiny = Math.pow(2.0,-966.0);
    while (p > 0) {
      int k;
      
      // Here is where a test for too many iterations would go.
      
      // This section of the program inspects for
      // negligible elements in the s and e arrays.  On
      // completion the variables kase and k are set as follows.
      
      // kase = 1     if s(p) and e[k-1] are negligible and k<p
      // kase = 2     if s(k) is negligible and k<p
      // kase = 3     if e[k-1] is negligible, k<p, and
      //              s(k), ..., s(p) are not negligible (qr step).
      // kase = 4     if e(p-1) is negligible (convergence).
      
      for (k = p - 2; k >= -1; k--) {
        if (k == -1) {
          break;
        }
        if (Math.abs(e[k]) <= tiny +eps * (Math.abs(s[k]) + Math.abs(s[k + 1]))) {
          e[k] = 0.0;
          break;
        }
      }
      int kase;
      if (k == p - 2) {
        kase = 4;
      } else {
        int ks;
        for (ks = p - 1; ks >= k; ks--) {
          if (ks == k) {
            break;
          }
          double t =
            (ks != p ?  Math.abs(e[ks]) : 0.) +
            (ks != k + 1 ?  Math.abs(e[ks-1]) : 0.);
          if (Math.abs(s[ks]) <= tiny + eps * t) {
            s[ks] = 0.0;
            break;
          }
        }
        if (ks == k) {
          kase = 3;
        } else if (ks == p - 1) {
          kase = 1;
        } else {
          kase = 2;
          k = ks;
        }
      }
      k++;
      
      // Perform the task indicated by kase.
      
      switch (kase) {
        
        // Deflate negligible s(p).
        
        case 1: {
          double f = e[p - 2];
          e[p - 2] = 0.0;
          for (int j = p - 2; j >= k; j--) {
            double t = Algebra.hypot(s[j], f);
            double cs = s[j] / t;
            double sn = f / t;
            s[j] = t;
            if (j != k) {
              f = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            if (wantv) {
              for (int i = 0; i < n; i++) {
                t = cs * v[i][j] + sn * v[i][p - 1];
                v[i][p - 1] = -sn * v[i][j] + cs * v[i][p - 1];
                v[i][j] = t;
              }
            }
          }
        }
          break;
        
        // Split at negligible s(k).
        
        case 2: {
          double f = e[k - 1];
          e[k - 1] = 0.0;
          for (int j = k; j < p; j++) {
            double t = Algebra.hypot(s[j], f);
            double cs = s[j] / t;
            double sn = f / t;
            s[j] = t;
            f = -sn * e[j];
            e[j] = cs * e[j];
            if (wantu) {
              for (int i = 0; i < m; i++) {
                t = cs * u[i][j] + sn * u[i][k - 1];
                u[i][k - 1] = -sn * u[i][j] + cs * u[i][k - 1];
                u[i][j] = t;
              }
            }
          }
        }
          break;
        
        // Perform one qr step.
        
        case 3: {
          
          // Calculate the shift.
          
          double scale = Math.max(Math.max(Math.max(Math.max(
              Math.abs(s[p - 1]), Math.abs(s[p - 2])), Math.abs(e[p - 2])),
              Math.abs(s[k])), Math.abs(e[k]));
          double sp = s[p - 1] / scale;
          double spm1 = s[p - 2] / scale;
          double epm1 = e[p - 2] / scale;
          double sk = s[k] / scale;
          double ek = e[k] / scale;
          double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
          double c = sp * epm1 * sp * epm1;
          double shift = 0.0;
          if (b != 0.0 || c != 0.0) {
            shift = Math.sqrt(b * b + c);
            if (b < 0.0) {
              shift = -shift;
            }
            shift = c / (b + shift);
          }
          double f = (sk + sp) * (sk - sp) + shift;
          double g = sk * ek;
          
          // Chase zeros.
          
          for (int j = k; j < p - 1; j++) {
            double t = Algebra.hypot(f, g);
            double cs = f / t;
            double sn = g / t;
            if (j != k) {
              e[j - 1] = t;
            }
            f = cs * s[j] + sn * e[j];
            e[j] = cs * e[j] - sn * s[j];
            g = sn * s[j + 1];
            s[j + 1] = cs * s[j + 1];
            if (wantv) {
              for (int i = 0; i < n; i++) {
                t = cs * v[i][j] + sn * v[i][j + 1];
                v[i][j + 1] = -sn * v[i][j] + cs * v[i][j + 1];
                v[i][j] = t;
              }
            }
            t = Algebra.hypot(f, g);
            cs = f / t;
            sn = g / t;
            s[j] = t;
            f = cs * e[j] + sn * s[j + 1];
            s[j + 1] = -sn * e[j] + cs * s[j + 1];
            g = sn * e[j + 1];
            e[j + 1] = cs * e[j + 1];
            if (wantu && j < m - 1) {
              for (int i = 0; i < m; i++) {
                t = cs * u[i][j] + sn * u[i][j + 1];
                u[i][j + 1] = -sn * u[i][j] + cs * u[i][j + 1];
                u[i][j] = t;
              }
            }
          }
          e[p - 2] = f;
          iter = iter + 1;
        }
          break;
        
        // Convergence.
        
        case 4: {
          
          // Make the singular values positive.
          
          if (s[k] <= 0.0) {
            s[k] = s[k] < 0.0 ? -s[k] : 0.0;
            if (wantv) {
              for (int i = 0; i <= pp; i++) {
                v[i][k] = -v[i][k];
              }
            }
          }
          
          // Order the singular values.
          
          while (k < pp) {
            if (s[k] >= s[k + 1]) {
              break;
            }
            double t = s[k];
            s[k] = s[k + 1];
            s[k + 1] = t;
            if (wantv && k < n - 1) {
              for (int i = 0; i < n; i++) {
                t = v[i][k + 1];
                v[i][k + 1] = v[i][k];
                v[i][k] = t;
              }
            }
            if (wantu && k < m - 1) {
              for (int i = 0; i < m; i++) {
                t = u[i][k + 1];
                u[i][k + 1] = u[i][k];
                u[i][k] = t;
              }
            }
            k++;
          }
          iter = 0;
          p--;
        }
          break;
        default:
          throw new IllegalStateException();
      }
    }
  }
  
  /**
   * Returns the two norm condition number, which is <tt>max(S) / min(S)</tt>.
   */
  public double cond() {
    return s[0] / s[Math.min(m, n) - 1];
  }
  
  /**
   * @return the diagonal matrix of singular values.
   */
  public Matrix getS() {
    double[][] s = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        s[i][j] = 0.0;
      }
      s[i][i] = this.s[i];
    }
    
    return new DenseMatrix(s);
  }
  
  /**
   * Returns the diagonal of <tt>S</tt>, which is a one-dimensional array of
   * singular values
   * 
   * @return diagonal of <tt>S</tt>.
   */
  public double[] getSingularValues() {
    return s;
  }
  
  /**
   * Returns the left singular vectors <tt>U</tt>.
   * 
   * @return <tt>U</tt>
   */
  public Matrix getU() {
    if (transpositionNeeded) { //case numRows() < numCols()
      return new DenseMatrix(v);
    } else {
      int numCols = Math.min(m + 1, n);
      Matrix r = new DenseMatrix(m, numCols);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < numCols; j++) {
          r.set(i, j, u[i][j]);
        }
      }

      return r;
    }
  }
  
  /**
   * Returns the right singular vectors <tt>V</tt>.
   * 
   * @return <tt>V</tt>
   */
  public Matrix getV() {
    if (transpositionNeeded) { //case numRows() < numCols()
      int numCols = Math.min(m + 1, n);
      Matrix r = new DenseMatrix(m, numCols);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < numCols; j++) {
          r.set(i, j, u[i][j]);
        }
      }

      return r;
    } else {
      return new DenseMatrix(v);
    }
  }
  
  /** Returns the two norm, which is <tt>max(S)</tt>. */
  public double norm2() {
    return s[0];
  }
  
  /**
   * Returns the effective numerical matrix rank, which is the number of
   * nonnegligible singular values.
   */
  public int rank() {
    double eps = Math.pow(2.0, -52.0);
    double tol = Math.max(m, n) * s[0] * eps;
    int r = 0;
    for (double value : s) {
      if (value > tol) {
        r++;
      }
    }
    return r;
  }
  
  /**
   * @parameter minSingularValue
   * minSingularValue - value below which singular values are ignored (a 0 or negative
   * value implies all singular value will be used)
   * @return Returns the n × n covariance matrix.
   * The covariance matrix is V × J × Vt where J is the diagonal matrix of the inverse
   *  of the squares of the singular values.
   */
  Matrix getCovariance(double minSingularValue) {
    Matrix j = new DenseMatrix(s.length,s.length);
    Matrix vMat = new DenseMatrix(this.v);
    for (int i = 0; i < s.length; i++) {
      j.set(i, i, s[i] >= minSingularValue ? 1 / (s[i] * s[i]) : 0.0);
    }
    return vMat.times(j).times(vMat.transpose());
  }
  
  /**
   * Returns a String with (propertyName, propertyValue) pairs. Useful for
   * debugging or to quickly get the rough picture. For example,
   * 
   * <pre>
   * rank          : 3
   * trace         : 0
   * </pre>
   */
  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append("---------------------------------------------------------------------\n");
    buf.append("SingularValueDecomposition(A) --> cond(A), rank(A), norm2(A), U, S, V\n");
    buf.append("---------------------------------------------------------------------\n");
    
    buf.append("cond = ");
    String unknown = "Illegal operation or error: ";
    try {
      buf.append(String.valueOf(this.cond()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    buf.append("\nrank = ");
    try {
      buf.append(String.valueOf(this.rank()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    buf.append("\nnorm2 = ");
    try {
      buf.append(String.valueOf(this.norm2()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    buf.append("\n\nU = ");
    try {
      buf.append(String.valueOf(this.getU()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    buf.append("\n\nS = ");
    try {
      buf.append(String.valueOf(this.getS()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    buf.append("\n\nV = ");
    try {
      buf.append(String.valueOf(this.getV()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }
    
    return buf.toString();
  }
}
