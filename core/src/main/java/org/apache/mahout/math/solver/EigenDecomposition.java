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

/**
 * Adapted from the public domain Jama code.
 */

package org.apache.mahout.math.solver;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

/**
 * Eigenvalues and eigenvectors of a real matrix.
 * <p/>
 * If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is diagonal and the eigenvector
 * matrix V is orthogonal. I.e. A = V.times(D.times(V.transpose())) and V.times(V.transpose())
 * equals the identity matrix.
 * <p/>
 * If A is not symmetric, then the eigenvalue matrix D is block diagonal with the real eigenvalues
 * in 1-by-1 blocks and any complex eigenvalues, lambda + i*mu, in 2-by-2 blocks, [lambda, mu; -mu,
 * lambda].  The columns of V represent the eigenvectors in the sense that A*V = V*D, i.e.
 * A.times(V) equals V.times(D).  The matrix V may be badly conditioned, or even singular, so the
 * validity of the equation A = V*D*inverse(V) depends upon V.cond().
 */
public class EigenDecomposition {

  /** Row and column dimension (square matrix). */
  private final int n;
  /** Arrays for internal storage of eigenvalues. */
  private final Vector d;
  private final Vector e;
  /** Array for internal storage of eigenvectors. */
  private final Matrix v;

  public EigenDecomposition(Matrix x) {
    this(x, isSymmetric(x));
  }

  public EigenDecomposition(Matrix x, boolean isSymmetric) {
    n = x.columnSize();
    d = new DenseVector(n);
    e = new DenseVector(n);
    v = new DenseMatrix(n, n);

    if (isSymmetric) {
      v.assign(x);

      // Tridiagonalize.
      tred2();

      // Diagonalize.
      tql2();

    } else {
      // Reduce to Hessenberg form.
      // Reduce Hessenberg to real Schur form.
      hqr2(orthes(x));
    }
  }

  /**
   * Return the eigenvector matrix
   *
   * @return V
   */
  public Matrix getV() {
    return v.like().assign(v);
  }

  /**
   * Return the real parts of the eigenvalues
   */
  public Vector getRealEigenvalues() {
    return d;
  }

  /**
   * Return the imaginary parts of the eigenvalues
   */
  public Vector getImagEigenvalues() {
    return e;
  }

  /**
   * Return the block diagonal eigenvalue matrix
   *
   * @return D
   */
  public Matrix getD() {
    Matrix x = new DenseMatrix(n, n);
    x.assign(0);
    x.viewDiagonal().assign(d);
    for (int i = 0; i < n; i++) {
      double v = e.getQuick(i);
      if (v > 0) {
        x.setQuick(i, i + 1, v);
      } else if (v < 0) {
        x.setQuick(i, i - 1, v);
      }
    }
    return x;
  }

  // Symmetric Householder reduction to tridiagonal form.
  private void tred2() {
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    d.assign(v.viewColumn(n - 1));

    // Householder reduction to tridiagonal form.

    for (int i = n - 1; i > 0; i--) {

      // Scale to avoid under/overflow.

      double scale = d.viewPart(0, i).norm(1);
      double h = 0.0;


      if (scale == 0.0) {
        e.setQuick(i, d.getQuick(i - 1));
        for (int j = 0; j < i; j++) {
          d.setQuick(j, v.getQuick(i - 1, j));
          v.setQuick(i, j, 0.0);
          v.setQuick(j, i, 0.0);
        }
      } else {

        // Generate Householder vector.

        for (int k = 0; k < i; k++) {
          d.setQuick(k, d.getQuick(k) / scale);
          h += d.getQuick(k) * d.getQuick(k);
        }
        double f = d.getQuick(i - 1);
        double g = Math.sqrt(h);
        if (f > 0) {
          g = -g;
        }
        e.setQuick(i, scale * g);
        h -= f * g;
        d.setQuick(i - 1, f - g);
        for (int j = 0; j < i; j++) {
          e.setQuick(j, 0.0);
        }

        // Apply similarity transformation to remaining columns.

        for (int j = 0; j < i; j++) {
          f = d.getQuick(j);
          v.setQuick(j, i, f);
          g = e.getQuick(j) + v.getQuick(j, j) * f;
          for (int k = j + 1; k <= i - 1; k++) {
            g += v.getQuick(k, j) * d.getQuick(k);
            e.setQuick(k, e.getQuick(k) + v.getQuick(k, j) * f);
          }
          e.setQuick(j, g);
        }
        f = 0.0;
        for (int j = 0; j < i; j++) {
          e.setQuick(j, e.getQuick(j) / h);
          f += e.getQuick(j) * d.getQuick(j);
        }
        double hh = f / (h + h);
        for (int j = 0; j < i; j++) {
          e.setQuick(j, e.getQuick(j) - hh * d.getQuick(j));
        }
        for (int j = 0; j < i; j++) {
          f = d.getQuick(j);
          g = e.getQuick(j);
          for (int k = j; k <= i - 1; k++) {
            v.setQuick(k, j, v.getQuick(k, j) - (f * e.getQuick(k) + g * d.getQuick(k)));
          }
          d.setQuick(j, v.getQuick(i - 1, j));
          v.setQuick(i, j, 0.0);
        }
      }
      d.setQuick(i, h);
    }

    // Accumulate transformations.

    for (int i = 0; i < n - 1; i++) {
      v.setQuick(n - 1, i, v.getQuick(i, i));
      v.setQuick(i, i, 1.0);
      double h = d.getQuick(i + 1);
      if (h != 0.0) {
        for (int k = 0; k <= i; k++) {
          d.setQuick(k, v.getQuick(k, i + 1) / h);
        }
        for (int j = 0; j <= i; j++) {
          double g = 0.0;
          for (int k = 0; k <= i; k++) {
            g += v.getQuick(k, i + 1) * v.getQuick(k, j);
          }
          for (int k = 0; k <= i; k++) {
            v.setQuick(k, j, v.getQuick(k, j) - g * d.getQuick(k));
          }
        }
      }
      for (int k = 0; k <= i; k++) {
        v.setQuick(k, i + 1, 0.0);
      }
    }
    d.assign(v.viewRow(n - 1));
    v.viewRow(n - 1).assign(0);
    v.setQuick(n - 1, n - 1, 1.0);
    e.setQuick(0, 0.0);
  }

  // Symmetric tridiagonal QL algorithm.
  private void tql2() {

    //  This is derived from the Algol procedures tql2, by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    e.viewPart(0, n - 1).assign(e.viewPart(1, n - 1));
    e.setQuick(n - 1, 0.0);

    double f = 0.0;
    double tst1 = 0.0;
    double eps = Math.pow(2.0, -52.0);
    for (int l = 0; l < n; l++) {

      // Find small subdiagonal element

      tst1 = Math.max(tst1, Math.abs(d.getQuick(l)) + Math.abs(e.getQuick(l)));
      int m = l;
      while (m < n) {
        if (Math.abs(e.getQuick(m)) <= eps * tst1) {
          break;
        }
        m++;
      }

      // If m == l, d.getQuick(l) is an eigenvalue,
      // otherwise, iterate.

      if (m > l) {
        do {
          // Compute implicit shift

          double g = d.getQuick(l);
          double p = (d.getQuick(l + 1) - g) / (2.0 * e.getQuick(l));
          double r = Math.hypot(p, 1.0);
          if (p < 0) {
            r = -r;
          }
          d.setQuick(l, e.getQuick(l) / (p + r));
          d.setQuick(l + 1, e.getQuick(l) * (p + r));
          double dl1 = d.getQuick(l + 1);
          double h = g - d.getQuick(l);
          for (int i = l + 2; i < n; i++) {
            d.setQuick(i, d.getQuick(i) - h);
          }
          f += h;

          // Implicit QL transformation.

          p = d.getQuick(m);
          double c = 1.0;
          double c2 = c;
          double c3 = c;
          double el1 = e.getQuick(l + 1);
          double s = 0.0;
          double s2 = 0.0;
          for (int i = m - 1; i >= l; i--) {
            c3 = c2;
            c2 = c;
            s2 = s;
            g = c * e.getQuick(i);
            h = c * p;
            r = Math.hypot(p, e.getQuick(i));
            e.setQuick(i + 1, s * r);
            s = e.getQuick(i) / r;
            c = p / r;
            p = c * d.getQuick(i) - s * g;
            d.setQuick(i + 1, h + s * (c * g + s * d.getQuick(i)));

            // Accumulate transformation.

            for (int k = 0; k < n; k++) {
              h = v.getQuick(k, i + 1);
              v.setQuick(k, i + 1, s * v.getQuick(k, i) + c * h);
              v.setQuick(k, i, c * v.getQuick(k, i) - s * h);
            }
          }
          p = -s * s2 * c3 * el1 * e.getQuick(l) / dl1;
          e.setQuick(l, s * p);
          d.setQuick(l, c * p);

          // Check for convergence.

        } while (Math.abs(e.getQuick(l)) > eps * tst1);
      }
      d.setQuick(l, d.getQuick(l) + f);
      e.setQuick(l, 0.0);
    }

    // Sort eigenvalues and corresponding vectors.

    for (int i = 0; i < n - 1; i++) {
      int k = i;
      double p = d.getQuick(i);
      for (int j = i + 1; j < n; j++) {
        if (d.getQuick(j) > p) {
          k = j;
          p = d.getQuick(j);
        }
      }
      if (k != i) {
        d.setQuick(k, d.getQuick(i));
        d.setQuick(i, p);
        for (int j = 0; j < n; j++) {
          p = v.getQuick(j, i);
          v.setQuick(j, i, v.getQuick(j, k));
          v.setQuick(j, k, p);
        }
      }
    }
  }

  // Nonsymmetric reduction to Hessenberg form.
  private Matrix orthes(Matrix x) {
    // Working storage for nonsymmetric algorithm.
    Vector ort = new DenseVector(n);
    Matrix hessenBerg = new DenseMatrix(n, n).assign(x);

    //  This is derived from the Algol procedures orthes and ortran,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutines in EISPACK.

    int low = 0;
    int high = n - 1;

    for (int m = low + 1; m <= high - 1; m++) {

      // Scale column.

      Vector hColumn = hessenBerg.viewColumn(m - 1).viewPart(m, high - m + 1);
      double scale = hColumn.norm(1);

      if (scale != 0.0) {
        // Compute Householder transformation.

        ort.viewPart(m, high - m + 1).assign(hColumn, Functions.plusMult(1 / scale));
        double h = ort.viewPart(m, high - m + 1).getLengthSquared();

        double g = Math.sqrt(h);
        if (ort.getQuick(m) > 0) {
          g = -g;
        }
        h -= ort.getQuick(m) * g;
        ort.setQuick(m, ort.getQuick(m) - g);

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)

        Vector ortPiece = ort.viewPart(m, high - m + 1);
        for (int j = m; j < n; j++) {
          double f = ortPiece.dot(hessenBerg.viewColumn(j).viewPart(m, high - m + 1)) / h;
          hessenBerg.viewColumn(j).viewPart(m, high - m + 1).assign(ortPiece, Functions.plusMult(-f));
        }

        for (int i = 0; i <= high; i++) {
          double f = ortPiece.dot(hessenBerg.viewRow(i).viewPart(m, high - m + 1)) / h;
          hessenBerg.viewRow(i).viewPart(m, high - m + 1).assign(ortPiece, Functions.plusMult(-f));
        }
        ort.setQuick(m, scale * ort.getQuick(m));
        hessenBerg.setQuick(m, m - 1, scale * g);
      }
    }

    // Accumulate transformations (Algol's ortran).

    v.assign(0);
    v.viewDiagonal().assign(1);

    for (int m = high - 1; m >= low + 1; m--) {
      if (hessenBerg.getQuick(m, m - 1) != 0.0) {
        ort.viewPart(m + 1, high - m).assign(hessenBerg.viewColumn(m - 1).viewPart(m + 1, high - m));
        for (int j = m; j <= high; j++) {
          double g = ort.viewPart(m, high - m + 1).dot(v.viewColumn(j).viewPart(m, high - m + 1));
          // Double division avoids possible underflow
          g = g / ort.getQuick(m) / hessenBerg.getQuick(m, m - 1);
          v.viewColumn(j).viewPart(m, high - m + 1).assign(ort.viewPart(m, high - m + 1), Functions.plusMult(g));
        }
      }
    }
    return hessenBerg;
  }


  // Complex scalar division.
  private double cdivr;
  private double cdivi;

  private void cdiv(double xr, double xi, double yr, double yi) {
    double r;
    double d;
    if (Math.abs(yr) > Math.abs(yi)) {
      r = yi / yr;
      d = yr + r * yi;
      cdivr = (xr + r * xi) / d;
      cdivi = (xi - r * xr) / d;
    } else {
      r = yr / yi;
      d = yi + r * yr;
      cdivr = (r * xr + xi) / d;
      cdivi = (r * xi - xr) / d;
    }
  }


  // Nonsymmetric reduction from Hessenberg to real Schur form.

  private void hqr2(Matrix h) {

    //  This is derived from the Algol procedure hqr2,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    // Initialize

    int nn = this.n;
    int n = nn - 1;
    int low = 0;
    int high = nn - 1;
    double eps = Math.pow(2.0, -52.0);
    double exshift = 0.0;
    double p = 0;
    double q = 0;
    double r = 0;
    double s = 0;
    double z = 0;
    double w;
    double x;
    double y;

    // Store roots isolated by balanc and compute matrix norm

    double norm = h.aggregate(Functions.PLUS, Functions.ABS);

    // Outer loop over eigenvalue index

    int iter = 0;
    while (n >= low) {

      // Look for single small sub-diagonal element

      int l = n;
      while (l > low) {
        s = Math.abs(h.getQuick(l - 1, l - 1)) + Math.abs(h.getQuick(l, l));
        if (s == 0.0) {
          s = norm;
        }
        if (Math.abs(h.getQuick(l, l - 1)) < eps * s) {
          break;
        }
        l--;
      }

      // Check for convergence

      if (l == n) {
        // One root found
        h.setQuick(n, n, h.getQuick(n, n) + exshift);
        d.setQuick(n, h.getQuick(n, n));
        e.setQuick(n, 0.0);
        n--;
        iter = 0;


      } else if (l == n - 1) {
        // Two roots found
        w = h.getQuick(n, n - 1) * h.getQuick(n - 1, n);
        p = (h.getQuick(n - 1, n - 1) - h.getQuick(n, n)) / 2.0;
        q = p * p + w;
        z = Math.sqrt(Math.abs(q));
        h.setQuick(n, n, h.getQuick(n, n) + exshift);
        h.setQuick(n - 1, n - 1, h.getQuick(n - 1, n - 1) + exshift);
        x = h.getQuick(n, n);

        // Real pair
        if (q >= 0) {
          if (p >= 0) {
            z = p + z;
          } else {
            z = p - z;
          }
          d.setQuick(n - 1, x + z);
          d.setQuick(n, d.getQuick(n - 1));
          if (z != 0.0) {
            d.setQuick(n, x - w / z);
          }
          e.setQuick(n - 1, 0.0);
          e.setQuick(n, 0.0);
          x = h.getQuick(n, n - 1);
          s = Math.abs(x) + Math.abs(z);
          p = x / s;
          q = z / s;
          r = Math.sqrt(p * p + q * q);
          p /= r;
          q /= r;

          // Row modification

          for (int j = n - 1; j < nn; j++) {
            z = h.getQuick(n - 1, j);
            h.setQuick(n - 1, j, q * z + p * h.getQuick(n, j));
            h.setQuick(n, j, q * h.getQuick(n, j) - p * z);
          }

          // Column modification

          for (int i = 0; i <= n; i++) {
            z = h.getQuick(i, n - 1);
            h.setQuick(i, n - 1, q * z + p * h.getQuick(i, n));
            h.setQuick(i, n, q * h.getQuick(i, n) - p * z);
          }

          // Accumulate transformations

          for (int i = low; i <= high; i++) {
            z = v.getQuick(i, n - 1);
            v.setQuick(i, n - 1, q * z + p * v.getQuick(i, n));
            v.setQuick(i, n, q * v.getQuick(i, n) - p * z);
          }

          // Complex pair

        } else {
          d.setQuick(n - 1, x + p);
          d.setQuick(n, x + p);
          e.setQuick(n - 1, z);
          e.setQuick(n, -z);
        }
        n -= 2;
        iter = 0;

        // No convergence yet

      } else {

        // Form shift

        x = h.getQuick(n, n);
        y = 0.0;
        w = 0.0;
        if (l < n) {
          y = h.getQuick(n - 1, n - 1);
          w = h.getQuick(n, n - 1) * h.getQuick(n - 1, n);
        }

        // Wilkinson's original ad hoc shift

        if (iter == 10) {
          exshift += x;
          for (int i = low; i <= n; i++) {
            h.setQuick(i, i, x);
          }
          s = Math.abs(h.getQuick(n, n - 1)) + Math.abs(h.getQuick(n - 1, n - 2));
          x = y = 0.75 * s;
          w = -0.4375 * s * s;
        }

        // MATLAB's new ad hoc shift

        if (iter == 30) {
          s = (y - x) / 2.0;
          s = s * s + w;
          if (s > 0) {
            s = Math.sqrt(s);
            if (y < x) {
              s = -s;
            }
            s = x - w / ((y - x) / 2.0 + s);
            for (int i = low; i <= n; i++) {
              h.setQuick(i, i, h.getQuick(i, i) - s);
            }
            exshift += s;
            x = y = w = 0.964;
          }
        }

        iter++;   // (Could check iteration count here.)

        // Look for two consecutive small sub-diagonal elements

        int m = n - 2;
        while (m >= l) {
          z = h.getQuick(m, m);
          r = x - z;
          s = y - z;
          p = (r * s - w) / h.getQuick(m + 1, m) + h.getQuick(m, m + 1);
          q = h.getQuick(m + 1, m + 1) - z - r - s;
          r = h.getQuick(m + 2, m + 1);
          s = Math.abs(p) + Math.abs(q) + Math.abs(r);
          p /= s;
          q /= s;
          r /= s;
          if (m == l) {
            break;
          }
          double hmag = Math.abs(h.getQuick(m - 1, m - 1)) + Math.abs(h.getQuick(m + 1, m + 1));
          double threshold = eps * Math.abs(p) * (Math.abs(z) + hmag);
          if (Math.abs(h.getQuick(m, m - 1)) * (Math.abs(q) + Math.abs(r)) < threshold) {
            break;
          }
          m--;
        }

        for (int i = m + 2; i <= n; i++) {
          h.setQuick(i, i - 2, 0.0);
          if (i > m + 2) {
            h.setQuick(i, i - 3, 0.0);
          }
        }

        // Double QR step involving rows l:n and columns m:n

        for (int k = m; k <= n - 1; k++) {
          boolean notlast = k != n - 1;
          if (k != m) {
            p = h.getQuick(k, k - 1);
            q = h.getQuick(k + 1, k - 1);
            r = notlast ? h.getQuick(k + 2, k - 1) : 0.0;
            x = Math.abs(p) + Math.abs(q) + Math.abs(r);
            if (x != 0.0) {
              p /= x;
              q /= x;
              r /= x;
            }
          }
          if (x == 0.0) {
            break;
          }
          s = Math.sqrt(p * p + q * q + r * r);
          if (p < 0) {
            s = -s;
          }
          if (s != 0) {
            if (k != m) {
              h.setQuick(k, k - 1, -s * x);
            } else if (l != m) {
              h.setQuick(k, k - 1, -h.getQuick(k, k - 1));
            }
            p += s;
            x = p / s;
            y = q / s;
            z = r / s;
            q /= p;
            r /= p;

            // Row modification

            for (int j = k; j < nn; j++) {
              p = h.getQuick(k, j) + q * h.getQuick(k + 1, j);
              if (notlast) {
                p += r * h.getQuick(k + 2, j);
                h.setQuick(k + 2, j, h.getQuick(k + 2, j) - p * z);
              }
              h.setQuick(k, j, h.getQuick(k, j) - p * x);
              h.setQuick(k + 1, j, h.getQuick(k + 1, j) - p * y);
            }

            // Column modification

            for (int i = 0; i <= Math.min(n, k + 3); i++) {
              p = x * h.getQuick(i, k) + y * h.getQuick(i, k + 1);
              if (notlast) {
                p += z * h.getQuick(i, k + 2);
                h.setQuick(i, k + 2, h.getQuick(i, k + 2) - p * r);
              }
              h.setQuick(i, k, h.getQuick(i, k) - p);
              h.setQuick(i, k + 1, h.getQuick(i, k + 1) - p * q);
            }

            // Accumulate transformations

            for (int i = low; i <= high; i++) {
              p = x * v.getQuick(i, k) + y * v.getQuick(i, k + 1);
              if (notlast) {
                p += z * v.getQuick(i, k + 2);
                v.setQuick(i, k + 2, v.getQuick(i, k + 2) - p * r);
              }
              v.setQuick(i, k, v.getQuick(i, k) - p);
              v.setQuick(i, k + 1, v.getQuick(i, k + 1) - p * q);
            }
          }  // (s != 0)
        }  // k loop
      }  // check convergence
    }  // while (n >= low)

    // Backsubstitute to find vectors of upper triangular form

    if (norm == 0.0) {
      return;
    }

    for (n = nn - 1; n >= 0; n--) {
      p = d.getQuick(n);
      q = e.getQuick(n);

      // Real vector

      double t;
      if (q == 0) {
        int l = n;
        h.setQuick(n, n, 1.0);
        for (int i = n - 1; i >= 0; i--) {
          w = h.getQuick(i, i) - p;
          r = 0.0;
          for (int j = l; j <= n; j++) {
            r += h.getQuick(i, j) * h.getQuick(j, n);
          }
          if (e.getQuick(i) < 0.0) {
            z = w;
            s = r;
          } else {
            l = i;
            if (e.getQuick(i) == 0.0) {
              if (w == 0.0) {
                h.setQuick(i, n, -r / (eps * norm));
              } else {
                h.setQuick(i, n, -r / w);
              }

              // Solve real equations

            } else {
              x = h.getQuick(i, i + 1);
              y = h.getQuick(i + 1, i);
              q = (d.getQuick(i) - p) * (d.getQuick(i) - p) + e.getQuick(i) * e.getQuick(i);
              t = (x * s - z * r) / q;
              h.setQuick(i, n, t);
              if (Math.abs(x) > Math.abs(z)) {
                h.setQuick(i + 1, n, (-r - w * t) / x);
              } else {
                h.setQuick(i + 1, n, (-s - y * t) / z);
              }
            }

            // Overflow control

            t = Math.abs(h.getQuick(i, n));
            if (eps * t * t > 1) {
              for (int j = i; j <= n; j++) {
                h.setQuick(j, n, h.getQuick(j, n) / t);
              }
            }
          }
        }

        // Complex vector

      } else if (q < 0) {
        int l = n - 1;

        // Last vector component imaginary so matrix is triangular

        if (Math.abs(h.getQuick(n, n - 1)) > Math.abs(h.getQuick(n - 1, n))) {
          h.setQuick(n - 1, n - 1, q / h.getQuick(n, n - 1));
          h.setQuick(n - 1, n, -(h.getQuick(n, n) - p) / h.getQuick(n, n - 1));
        } else {
          cdiv(0.0, -h.getQuick(n - 1, n), h.getQuick(n - 1, n - 1) - p, q);
          h.setQuick(n - 1, n - 1, cdivr);
          h.setQuick(n - 1, n, cdivi);
        }
        h.setQuick(n, n - 1, 0.0);
        h.setQuick(n, n, 1.0);
        for (int i = n - 2; i >= 0; i--) {
          double ra = 0.0;
          double sa = 0.0;
          for (int j = l; j <= n; j++) {
            ra += h.getQuick(i, j) * h.getQuick(j, n - 1);
            sa += h.getQuick(i, j) * h.getQuick(j, n);
          }
          w = h.getQuick(i, i) - p;

          if (e.getQuick(i) < 0.0) {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if (e.getQuick(i) == 0) {
              cdiv(-ra, -sa, w, q);
              h.setQuick(i, n - 1, cdivr);
              h.setQuick(i, n, cdivi);
            } else {

              // Solve complex equations

              x = h.getQuick(i, i + 1);
              y = h.getQuick(i + 1, i);
              double vr = (d.getQuick(i) - p) * (d.getQuick(i) - p) + e.getQuick(i) * e.getQuick(i) - q * q;
              double vi = (d.getQuick(i) - p) * 2.0 * q;
              if (vr == 0.0 && vi == 0.0) {
                double hmag = Math.abs(x) + Math.abs(y);
                vr = eps * norm * (Math.abs(w) + Math.abs(q) + hmag + Math.abs(z));
              }
              cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              h.setQuick(i, n - 1, cdivr);
              h.setQuick(i, n, cdivi);
              if (Math.abs(x) > (Math.abs(z) + Math.abs(q))) {
                h.setQuick(i + 1, n - 1, (-ra - w * h.getQuick(i, n - 1) + q * h.getQuick(i, n)) / x);
                h.setQuick(i + 1, n, (-sa - w * h.getQuick(i, n) - q * h.getQuick(i, n - 1)) / x);
              } else {
                cdiv(-r - y * h.getQuick(i, n - 1), -s - y * h.getQuick(i, n), z, q);
                h.setQuick(i + 1, n - 1, cdivr);
                h.setQuick(i + 1, n, cdivi);
              }
            }

            // Overflow control

            t = Math.max(Math.abs(h.getQuick(i, n - 1)), Math.abs(h.getQuick(i, n)));
            if (eps * t * t > 1) {
              for (int j = i; j <= n; j++) {
                h.setQuick(j, n - 1, h.getQuick(j, n - 1) / t);
                h.setQuick(j, n, h.getQuick(j, n) / t);
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots

    for (int i = 0; i < nn; i++) {
      if (i < low || i > high) {
        for (int j = i; j < nn; j++) {
          v.setQuick(i, j, h.getQuick(i, j));
        }
      }
    }

    // Back transformation to get eigenvectors of original matrix

    for (int j = nn - 1; j >= low; j--) {
      for (int i = low; i <= high; i++) {
        z = 0.0;
        for (int k = low; k <= Math.min(j, high); k++) {
          z += v.getQuick(i, k) * h.getQuick(k, j);
        }
        v.setQuick(i, j, z);
      }
    }
  }

  private static boolean isSymmetric(Matrix a) {
    /*
    Symmetry flag.
    */
    int n = a.columnSize();

    boolean isSymmetric = true;
    for (int j = 0; (j < n) && isSymmetric; j++) {
      for (int i = 0; (i < n) && isSymmetric; i++) {
        isSymmetric = a.getQuick(i, j) == a.getQuick(j, i);
      }
    }
    return isSymmetric;
  }
}
