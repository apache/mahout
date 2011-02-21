/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix1D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;

import java.io.Serializable;

import static org.apache.mahout.math.Algebra.hypot;
import static org.apache.mahout.math.matrix.linalg.Property.*;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class EigenvalueDecomposition implements Serializable {

  /** Row and column dimension (square matrix). */
  private final int n;

  /** Arrays for internal storage of eigenvalues. */
  private final double[] d;
  private final double[] e;

  /** Array for internal storage of eigenvectors. */
  private final double[][] V;

  /** Array for internal storage of nonsymmetric Hessenberg form. */
  private double[][] H;

  /** Working storage for nonsymmetric algorithm. */
  private double[] ort;

  // Complex scalar division.

  private double cdivr;
  private double cdivi;

  /**
   * Constructs and returns a new eigenvalue decomposition object; The decomposed matrices can be retrieved via instance
   * methods of the returned decomposition object. Checks for symmetry, then constructs the eigenvalue decomposition.
   *
   * @param A A square matrix.
   * @throws IllegalArgumentException if <tt>A</tt> is not square.
   */
  public EigenvalueDecomposition(DoubleMatrix2D A) {
    checkSquare(A);

    n = A.columns();
    V = new double[n][n];
    d = new double[n];
    e = new double[n];

    boolean issymmetric = DEFAULT.isSymmetric(A);

    if (issymmetric) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          V[i][j] = A.getQuick(i, j);
        }
      }

      // Tridiagonalize.
      tred2();

      // Diagonalize.
      tql2();

    } else {
      H = new double[n][n];
      ort = new double[n];

      for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
          H[i][j] = A.getQuick(i, j);
        }
      }

      // Reduce to Hessenberg form.
      orthes();

      // Reduce Hessenberg to real Schur form.
      hqr2();
    }
  }

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

  /**
   * Returns the block diagonal eigenvalue matrix, <tt>D</tt>.
   *
   * @return <tt>D</tt>
   */
  public DoubleMatrix2D getD() {
    double[][] D = new double[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        D[i][j] = 0.0;
      }
      D[i][i] = d[i];
      if (e[i] > 0) {
        D[i][i + 1] = e[i];
      } else if (e[i] < 0) {
        D[i][i - 1] = e[i];
      }
    }
    return new DenseDoubleMatrix2D(D);
  }

  /**
   * Returns the imaginary parts of the eigenvalues.
   *
   * @return imag(diag(D))
   */
  public DoubleMatrix1D getImagEigenvalues() {
    return new DenseDoubleMatrix1D(e);
  }

  /**
   * Returns the real parts of the eigenvalues.
   *
   * @return real(diag(D))
   */
  public DoubleMatrix1D getRealEigenvalues() {
    return new DenseDoubleMatrix1D(d);
  }

  /**
   * Returns the eigenvector matrix, <tt>V</tt>
   *
   * @return <tt>V</tt>
   */
  public DoubleMatrix2D getV() {
    return new DenseDoubleMatrix2D(V);
  }

  /** Nonsymmetric reduction from Hessenberg to real Schur form. */
  private void hqr2() {
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

    // Store roots isolated by balanc and compute matrix norm

    double norm = 0.0;
    for (int i = 0; i < nn; i++) {
      if (i < low || i > high) {
        d[i] = H[i][i];
        e[i] = 0.0;
      }
      for (int j = Math.max(i - 1, 0); j < nn; j++) {
        norm += Math.abs(H[i][j]);
      }
    }

    // Outer loop over eigenvalue index

    int iter = 0;
    double y;
    double x;
    double w;
    double z = 0;
    double s = 0;
    double r = 0;
    double q = 0;
    double p = 0;
    double exshift = 0.0;
    while (n >= low) {

      // Look for single small sub-diagonal element

      int l = n;
      while (l > low) {
        s = Math.abs(H[l - 1][l - 1]) + Math.abs(H[l][l]);
        if (s == 0.0) {
          s = norm;
        }
        if (Math.abs(H[l][l - 1]) < eps * s) {
          break;
        }
        l--;
      }

      // Check for convergence
      // One root found

      if (l == n) {
        H[n][n] += exshift;
        d[n] = H[n][n];
        e[n] = 0.0;
        n--;
        iter = 0;

        // Two roots found

      } else if (l == n - 1) {
        w = H[n][n - 1] * H[n - 1][n];
        p = (H[n - 1][n - 1] - H[n][n]) / 2.0;
        q = p * p + w;
        z = Math.sqrt(Math.abs(q));
        H[n][n] += exshift;
        H[n - 1][n - 1] += exshift;
        x = H[n][n];

        // Real pair

        if (q >= 0) {
          if (p >= 0) {
            z = p + z;
          } else {
            z = p - z;
          }
          d[n - 1] = x + z;
          d[n] = d[n - 1];
          if (z != 0.0) {
            d[n] = x - w / z;
          }
          e[n - 1] = 0.0;
          e[n] = 0.0;
          x = H[n][n - 1];
          s = Math.abs(x) + Math.abs(z);
          p = x / s;
          q = z / s;
          r = Math.sqrt(p * p + q * q);
          p /= r;
          q /= r;

          // Row modification

          for (int j = n - 1; j < nn; j++) {
            z = H[n - 1][j];
            H[n - 1][j] = q * z + p * H[n][j];
            H[n][j] = q * H[n][j] - p * z;
          }

          // Column modification

          for (int i = 0; i <= n; i++) {
            z = H[i][n - 1];
            H[i][n - 1] = q * z + p * H[i][n];
            H[i][n] = q * H[i][n] - p * z;
          }

          // Accumulate transformations

          for (int i = low; i <= high; i++) {
            z = V[i][n - 1];
            V[i][n - 1] = q * z + p * V[i][n];
            V[i][n] = q * V[i][n] - p * z;
          }

          // Complex pair

        } else {
          d[n - 1] = x + p;
          d[n] = x + p;
          e[n - 1] = z;
          e[n] = -z;
        }
        n -= 2;
        iter = 0;

        // No convergence yet

      } else {

        // Form shift

        x = H[n][n];
        y = 0.0;
        w = 0.0;
        if (l < n) {
          y = H[n - 1][n - 1];
          w = H[n][n - 1] * H[n - 1][n];
        }

        // Wilkinson's original ad hoc shift

        if (iter == 10) {
          exshift += x;
          for (int i = low; i <= n; i++) {
            H[i][i] -= x;
          }
          s = Math.abs(H[n][n - 1]) + Math.abs(H[n - 1][n - 2]);
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
              H[i][i] -= s;
            }
            exshift += s;
            x = y = w = 0.964;
          }
        }

        iter += 1;   // (Could check iteration count here.)

        // Look for two consecutive small sub-diagonal elements

        int m = n - 2;
        while (m >= l) {
          z = H[m][m];
          r = x - z;
          s = y - z;
          p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
          q = H[m + 1][m + 1] - z - r - s;
          r = H[m + 2][m + 1];
          s = Math.abs(p) + Math.abs(q) + Math.abs(r);
          p /= s;
          q /= s;
          r /= s;
          if (m == l) {
            break;
          }
          if (Math.abs(H[m][m - 1]) * (Math.abs(q) + Math.abs(r))
              < eps * (Math.abs(p) * (Math.abs(H[m - 1][m - 1]) + Math.abs(z) + Math.abs(H[m + 1][m + 1])))) {
            break;
          }
          m--;
        }

        for (int i = m + 2; i <= n; i++) {
          H[i][i - 2] = 0.0;
          if (i > m + 2) {
            H[i][i - 3] = 0.0;
          }
        }

        // Double QR step involving rows l:n and columns m:n

        for (int k = m; k <= n - 1; k++) {
          boolean notlast = k != n - 1;
          if (k != m) {
            p = H[k][k - 1];
            q = H[k + 1][k - 1];
            r = notlast ? H[k + 2][k - 1] : 0.0;
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
              H[k][k - 1] = -s * x;
            } else if (l != m) {
              H[k][k - 1] = -H[k][k - 1];
            }
            p += s;
            x = p / s;
            y = q / s;
            z = r / s;
            q /= p;
            r /= p;

            // Row modification

            for (int j = k; j < nn; j++) {
              p = H[k][j] + q * H[k + 1][j];
              if (notlast) {
                p += r * H[k + 2][j];
                H[k + 2][j] -= p * z;
              }
              H[k][j] -= p * x;
              H[k + 1][j] -= p * y;
            }

            // Column modification

            for (int i = 0; i <= Math.min(n, k + 3); i++) {
              p = x * H[i][k] + y * H[i][k + 1];
              if (notlast) {
                p += z * H[i][k + 2];
                H[i][k + 2] -= p * r;
              }
              H[i][k] -= p;
              H[i][k + 1] -= p * q;
            }

            // Accumulate transformations

            for (int i = low; i <= high; i++) {
              p = x * V[i][k] + y * V[i][k + 1];
              if (notlast) {
                p += z * V[i][k + 2];
                V[i][k + 2] -= p * r;
              }
              V[i][k] -= p;
              V[i][k + 1] -= p * q;
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
      p = d[n];
      q = e[n];

      // Real vector

      double t;
      if (q == 0) {
        int l = n;
        H[n][n] = 1.0;
        for (int i = n - 1; i >= 0; i--) {
          w = H[i][i] - p;
          r = 0.0;
          for (int j = l; j <= n; j++) {
            r += H[i][j] * H[j][n];
          }
          if (e[i] < 0.0) {
            z = w;
            s = r;
          } else {
            l = i;
            if (e[i] == 0.0) {
              if (w != 0.0) {
                H[i][n] = -r / w;
              } else {
                H[i][n] = -r / (eps * norm);
              }

              // Solve real equations

            } else {
              x = H[i][i + 1];
              y = H[i + 1][i];
              q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
              t = (x * s - z * r) / q;
              H[i][n] = t;
              if (Math.abs(x) > Math.abs(z)) {
                H[i + 1][n] = (-r - w * t) / x;
              } else {
                H[i + 1][n] = (-s - y * t) / z;
              }
            }

            // Overflow control

            t = Math.abs(H[i][n]);
            if ((eps * t) * t > 1) {
              for (int j = i; j <= n; j++) {
                H[j][n] /= t;
              }
            }
          }
        }

        // Complex vector

      } else if (q < 0) {
        int l = n - 1;

        // Last vector component imaginary so matrix is triangular

        if (Math.abs(H[n][n - 1]) > Math.abs(H[n - 1][n])) {
          H[n - 1][n - 1] = q / H[n][n - 1];
          H[n - 1][n] = -(H[n][n] - p) / H[n][n - 1];
        } else {
          cdiv(0.0, -H[n - 1][n], H[n - 1][n - 1] - p, q);
          H[n - 1][n - 1] = cdivr;
          H[n - 1][n] = cdivi;
        }
        H[n][n - 1] = 0.0;
        H[n][n] = 1.0;
        for (int i = n - 2; i >= 0; i--) {
          double ra = 0.0;
          double sa = 0.0;
          for (int j = l; j <= n; j++) {
            ra += H[i][j] * H[j][n - 1];
            sa += H[i][j] * H[j][n];
          }
          w = H[i][i] - p;

          if (e[i] < 0.0) {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if (e[i] == 0) {
              cdiv(-ra, -sa, w, q);
              H[i][n - 1] = cdivr;
              H[i][n] = cdivi;
            } else {

              // Solve complex equations

              x = H[i][i + 1];
              y = H[i + 1][i];
              double vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
              double vi = (d[i] - p) * 2.0 * q;
              if (vr == 0.0 && vi == 0.0) {
                vr = eps * norm * (Math.abs(w) + Math.abs(q) + Math.abs(x) + Math.abs(y) + Math.abs(z));
              }
              cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              H[i][n - 1] = cdivr;
              H[i][n] = cdivi;
              if (Math.abs(x) > (Math.abs(z) + Math.abs(q))) {
                H[i + 1][n - 1] = (-ra - w * H[i][n - 1] + q * H[i][n]) / x;
                H[i + 1][n] = (-sa - w * H[i][n] - q * H[i][n - 1]) / x;
              } else {
                cdiv(-r - y * H[i][n - 1], -s - y * H[i][n], z, q);
                H[i + 1][n - 1] = cdivr;
                H[i + 1][n] = cdivi;
              }
            }

            // Overflow control

            t = Math.max(Math.abs(H[i][n - 1]), Math.abs(H[i][n]));
            if ((eps * t) * t > 1) {
              for (int j = i; j <= n; j++) {
                H[j][n - 1] /= t;
                H[j][n] /= t;
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots

    for (int i = 0; i < nn; i++) {
      if (i < low || i > high) {
        System.arraycopy(H[i], i, V[i], i, nn - i);
      }
    }

    // Back transformation to get eigenvectors of original matrix

    for (int j = nn - 1; j >= low; j--) {
      for (int i = low; i <= high; i++) {
        z = 0.0;
        for (int k = low; k <= Math.min(j, high); k++) {
          z += V[i][k] * H[k][j];
        }
        V[i][j] = z;
      }
    }
  }

  /** Nonsymmetric reduction to Hessenberg form. */
  private void orthes() {
    //  This is derived from the Algol procedures orthes and ortran,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutines in EISPACK.

    int low = 0;
    int high = n - 1;

    for (int m = low + 1; m <= high - 1; m++) {

      // Scale column.

      double scale = 0.0;
      for (int i = m; i <= high; i++) {
        scale += Math.abs(H[i][m - 1]);
      }
      if (scale != 0.0) {

        // Compute Householder transformation.

        double h = 0.0;
        for (int i = high; i >= m; i--) {
          ort[i] = H[i][m - 1] / scale;
          h += ort[i] * ort[i];
        }
        double g = Math.sqrt(h);
        if (ort[m] > 0) {
          g = -g;
        }
        h -= ort[m] * g;
        ort[m] -= g;

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)

        for (int j = m; j < n; j++) {
          double f = 0.0;
          for (int i = high; i >= m; i--) {
            f += ort[i] * H[i][j];
          }
          f /= h;
          for (int i = m; i <= high; i++) {
            H[i][j] -= f * ort[i];
          }
        }

        for (int i = 0; i <= high; i++) {
          double f = 0.0;
          for (int j = high; j >= m; j--) {
            f += ort[j] * H[i][j];
          }
          f /= h;
          for (int j = m; j <= high; j++) {
            H[i][j] -= f * ort[j];
          }
        }
        ort[m] = scale * ort[m];
        H[m][m - 1] = scale * g;
      }
    }

    // Accumulate transformations (Algol's ortran).

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        V[i][j] = i == j ? 1.0 : 0.0;
      }
    }

    for (int m = high - 1; m >= low + 1; m--) {
      if (H[m][m - 1] != 0.0) {
        for (int i = m + 1; i <= high; i++) {
          ort[i] = H[i][m - 1];
        }
        for (int j = m; j <= high; j++) {
          double g = 0.0;
          for (int i = m; i <= high; i++) {
            g += ort[i] * V[i][j];
          }
          // Double division avoids possible underflow
          g = (g / ort[m]) / H[m][m - 1];
          for (int i = m; i <= high; i++) {
            V[i][j] += g * ort[i];
          }
        }
      }
    }
  }

  /**
   * Returns a String with (propertyName, propertyValue) pairs. Useful for debugging or to quickly get the rough
   * picture. For example,
   * <pre>
   * rank          : 3
   * trace         : 0
   * </pre>
   */
  public String toString() {
    StringBuilder buf = new StringBuilder();

    buf.append("---------------------------------------------------------------------\n");
    buf.append("EigenvalueDecomposition(A) --> D, V, realEigenvalues, imagEigenvalues\n");
    buf.append("---------------------------------------------------------------------\n");

    buf.append("realEigenvalues = ");
    String unknown = "Illegal operation or error: ";
    try {
      buf.append(String.valueOf(this.getRealEigenvalues()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\nimagEigenvalues = ");
    try {
      buf.append(String.valueOf(this.getImagEigenvalues()));
    } catch (IllegalArgumentException exc) {
      buf.append(unknown).append(exc.getMessage());
    }

    buf.append("\n\nD = ");
    try {
      buf.append(String.valueOf(this.getD()));
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

  /** Symmetric tridiagonal QL algorithm. */
  private void tql2() {

    //  This is derived from the Algol procedures tql2, by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    System.arraycopy(e, 1, e, 0, n - 1);
    e[n - 1] = 0.0;

    double f = 0.0;
    double tst1 = 0.0;
    double eps = Math.pow(2.0, -52.0);
    for (int l = 0; l < n; l++) {

      // Find small subdiagonal element

      tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
      int m = l;
      while (m < n) {
        if (Math.abs(e[m]) <= eps * tst1) {
          break;
        }
        m++;
      }

      // If m == l, d[l] is an eigenvalue,
      // otherwise, iterate.

      if (m > l) {
        int iter = 0;
        do {
          iter += 1;  // (Could check iteration count here.)

          // Compute implicit shift

          double g = d[l];
          double p = (d[l + 1] - g) / (2.0 * e[l]);
          double r = hypot(p, 1.0);
          if (p < 0) {
            r = -r;
          }
          d[l] = e[l] / (p + r);
          d[l + 1] = e[l] * (p + r);
          double dl1 = d[l + 1];
          double h = g - d[l];
          for (int i = l + 2; i < n; i++) {
            d[i] -= h;
          }
          f += h;

          // Implicit QL transformation.

          p = d[m];
          double c = 1.0;
          double c2 = c;
          double c3 = c;
          double el1 = e[l + 1];
          double s = 0.0;
          double s2 = 0.0;
          for (int i = m - 1; i >= l; i--) {
            c3 = c2;
            c2 = c;
            s2 = s;
            g = c * e[i];
            h = c * p;
            r = hypot(p, e[i]);
            e[i + 1] = s * r;
            s = e[i] / r;
            c = p / r;
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);

            // Accumulate transformation.

            for (int k = 0; k < n; k++) {
              h = V[k][i + 1];
              V[k][i + 1] = s * V[k][i] + c * h;
              V[k][i] = c * V[k][i] - s * h;
            }
          }
          p = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p;
          d[l] = c * p;

          // Check for convergence.

        } while (Math.abs(e[l]) > eps * tst1);
      }
      d[l] += f;
      e[l] = 0.0;
    }

    // Sort eigenvalues and corresponding vectors.

    for (int i = 0; i < n - 1; i++) {
      int k = i;
      double p = d[i];
      for (int j = i + 1; j < n; j++) {
        if (d[j] < p) {
          k = j;
          p = d[j];
        }
      }
      if (k != i) {
        d[k] = d[i];
        d[i] = p;
        for (int j = 0; j < n; j++) {
          p = V[j][i];
          V[j][i] = V[j][k];
          V[j][k] = p;
        }
      }
    }
  }

  /** Symmetric Householder reduction to tridiagonal form. */
  private void tred2() {
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.


    System.arraycopy(V[n - 1], 0, d, 0, n);


    // Householder reduction to tridiagonal form.

    for (int i = n - 1; i > 0; i--) {

      // Scale to avoid under/overflow.

      double scale = 0.0;
      for (int k = 0; k < i; k++) {
        scale += Math.abs(d[k]);
      }
      double h = 0.0;
      if (scale == 0.0) {
        e[i] = d[i - 1];
        for (int j = 0; j < i; j++) {
          d[j] = V[i - 1][j];
          V[i][j] = 0.0;
          V[j][i] = 0.0;
        }
      } else {

        // Generate Householder vector.

        for (int k = 0; k < i; k++) {
          d[k] /= scale;
          h += d[k] * d[k];
        }
        double f = d[i - 1];
        double g = Math.sqrt(h);
        if (f > 0) {
          g = -g;
        }
        e[i] = scale * g;
        h -= f * g;
        d[i - 1] = f - g;
        for (int j = 0; j < i; j++) {
          e[j] = 0.0;
        }

        // Apply similarity transformation to remaining columns.

        for (int j = 0; j < i; j++) {
          f = d[j];
          V[j][i] = f;
          g = e[j] + V[j][j] * f;
          for (int k = j + 1; k <= i - 1; k++) {
            g += V[k][j] * d[k];
            e[k] += V[k][j] * f;
          }
          e[j] = g;
        }
        f = 0.0;
        for (int j = 0; j < i; j++) {
          e[j] /= h;
          f += e[j] * d[j];
        }
        double hh = f / (h + h);
        for (int j = 0; j < i; j++) {
          e[j] -= hh * d[j];
        }
        for (int j = 0; j < i; j++) {
          f = d[j];
          g = e[j];
          for (int k = j; k <= i - 1; k++) {
            V[k][j] -= f * e[k] + g * d[k];
          }
          d[j] = V[i - 1][j];
          V[i][j] = 0.0;
        }
      }
      d[i] = h;
    }

    // Accumulate transformations.

    for (int i = 0; i < n - 1; i++) {
      V[n - 1][i] = V[i][i];
      V[i][i] = 1.0;
      double h = d[i + 1];
      if (h != 0.0) {
        for (int k = 0; k <= i; k++) {
          d[k] = V[k][i + 1] / h;
        }
        for (int j = 0; j <= i; j++) {
          double g = 0.0;
          for (int k = 0; k <= i; k++) {
            g += V[k][i + 1] * V[k][j];
          }
          for (int k = 0; k <= i; k++) {
            V[k][j] -= g * d[k];
          }
        }
      }
      for (int k = 0; k <= i; k++) {
        V[k][i + 1] = 0.0;
      }
    }
    for (int j = 0; j < n; j++) {
      d[j] = V[n - 1][j];
      V[n - 1][j] = 0.0;
    }
    V[n - 1][n - 1] = 1.0;
    e[0] = 0.0;
  }
}
