/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class SeqBlas implements Blas {

  public static final Blas seqBlas = new SeqBlas();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private SeqBlas() {
  }

  public void assign(DoubleMatrix2D A, UnaryFunction function) {
    A.assign(function);
  }

  public void assign(DoubleMatrix2D A, DoubleMatrix2D B,
                     BinaryFunction function) {
    A.assign(B, function);
  }

  public double dasum(DoubleMatrix1D x) {
    return x.aggregate(Functions.plus, Functions.abs);
  }

  public void daxpy(double alpha, DoubleMatrix1D x, DoubleMatrix1D y) {
    y.assign(x, Functions.plusMult(alpha));
  }

  public void daxpy(double alpha, DoubleMatrix2D A, DoubleMatrix2D B) {
    B.assign(A, Functions.plusMult(alpha));
  }

  public void dcopy(DoubleMatrix1D x, DoubleMatrix1D y) {
    y.assign(x);
  }

  public void dcopy(DoubleMatrix2D A, DoubleMatrix2D B) {
    B.assign(A);
  }

  public double ddot(DoubleMatrix1D x, DoubleMatrix1D y) {
    return x.zDotProduct(y);
  }

  public void dgemm(boolean transposeA, boolean transposeB, double alpha, DoubleMatrix2D A, DoubleMatrix2D B,
                    double beta, DoubleMatrix2D C) {
    A.zMult(B, C, alpha, beta, transposeA, transposeB);
  }

  public void dgemv(boolean transposeA, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta,
                    DoubleMatrix1D y) {
    A.zMult(x, y, alpha, beta, transposeA);
  }

  public void dger(double alpha, DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A) {
    PlusMult fun = PlusMult.plusMult(0);
    for (int i = A.rows(); --i >= 0;) {
      fun.setMultiplicator(alpha * x.getQuick(i));
      A.viewRow(i).assign(y, fun);

    }
  }

  public double dnrm2(DoubleMatrix1D x) {
    return Math.sqrt(Algebra.norm2(x));
  }

  public void drot(DoubleMatrix1D x, DoubleMatrix1D y, double c, double s) {
    x.checkSize(y);
    DoubleMatrix1D tmp = x.copy();

    x.assign(Functions.mult(c));
    x.assign(y, Functions.plusMult(s));

    y.assign(Functions.mult(c));
    y.assign(tmp, Functions.minusMult(s));
  }

  public void drotg(double a, double b, double[] rotvec) {

    double roe = b;

    if (Math.abs(a) > Math.abs(b)) {
      roe = a;
    }

    double scale = Math.abs(a) + Math.abs(b);

    double z;
    double r;
    double s;
    double c;
    if (scale != 0.0) {

      double ra = a / scale;
      double rb = b / scale;
      r = scale * Math.sqrt(ra * ra + rb * rb);
      r = sign(1.0, roe) * r;
      c = a / r;
      s = b / r;
      z = 1.0;
      if (Math.abs(a) > Math.abs(b)) {
        z = s;
      }
      if ((Math.abs(b) >= Math.abs(a)) && (c != 0.0)) {
        z = 1.0 / c;
      }

    } else {

      c = 1.0;
      s = 0.0;
      r = 0.0;
      z = 0.0;

    }

    a = r;
    b = z;

    rotvec[0] = a;
    rotvec[1] = b;
    rotvec[2] = c;
    rotvec[3] = s;

  }

  public void dscal(double alpha, DoubleMatrix1D x) {
    x.assign(Functions.mult(alpha));
  }

  public void dscal(double alpha, DoubleMatrix2D A) {
    A.assign(Functions.mult(alpha));
  }

  public void dswap(DoubleMatrix1D x, DoubleMatrix1D y) {
    y.swap(x);
  }

  public void dswap(DoubleMatrix2D A, DoubleMatrix2D B) {
    //B.swap(A); not yet implemented
    A.checkShape(B);
    for (int i = A.rows(); --i >= 0;) {
      A.viewRow(i).swap(B.viewRow(i));
    }
  }

  public void dsymv(boolean isUpperTriangular, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta,
                    DoubleMatrix1D y) {
    if (isUpperTriangular) {
      A = A.viewDice();
    }
    Property.checkSquare(A);
    int size = A.rows();
    if (size != x.size() && size != y.size()) {
      throw new IllegalArgumentException(A.toStringShort() + ", " + x.toStringShort() + ", " + y.toStringShort());
    }
    DoubleMatrix1D tmp = x.like();
    for (int i = 0; i < size; i++) {
      double sum = 0;
      for (int j = 0; j <= i; j++) {
        sum += A.getQuick(i, j) * x.getQuick(j);
      }
      for (int j = i + 1; j < size; j++) {
        sum += A.getQuick(j, i) * x.getQuick(j);
      }
      tmp.setQuick(i, alpha * sum + beta * y.getQuick(i));
    }
    y.assign(tmp);
  }

  public void dtrmv(boolean isUpperTriangular, boolean transposeA, boolean isUnitTriangular, DoubleMatrix2D A,
                    DoubleMatrix1D x) {
    if (transposeA) {
      A = A.viewDice();
      isUpperTriangular = !isUpperTriangular;
    }

    Property.checkSquare(A);
    int size = A.rows();
    if (size != x.size()) {
      throw new IllegalArgumentException(A.toStringShort() + ", " + x.toStringShort());
    }

    DoubleMatrix1D b = x.like();
    DoubleMatrix1D y = x.like();
    if (isUnitTriangular) {
      y.assign(1);
    } else {
      for (int i = 0; i < size; i++) {
        y.setQuick(i, A.getQuick(i, i));
      }
    }

    for (int i = 0; i < size; i++) {
      double sum = 0;
      if (!isUpperTriangular) {
        for (int j = 0; j < i; j++) {
          sum += A.getQuick(i, j) * x.getQuick(j);
        }
        sum += y.getQuick(i) * x.getQuick(i);
      } else {
        sum += y.getQuick(i) * x.getQuick(i);
        for (int j = i + 1; j < size; j++) {
          sum += A.getQuick(i, j) * x.getQuick(j);
        }
      }
      b.setQuick(i, sum);
    }
    x.assign(b);
  }

  public int idamax(DoubleMatrix1D x) {
    int maxIndex = -1;
    double maxValue = Double.MIN_VALUE;
    for (int i = x.size(); --i >= 0;) {
      double v = Math.abs(x.getQuick(i));
      if (v > maxValue) {
        maxValue = v;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  /**
   * Implements the FORTRAN sign (not sin) function. See the code for details.
   *
   * @param a a
   * @param b b
   */
  private static double sign(double a, double b) {
    return b < 0.0 ? -Math.abs(a) : Math.abs(a);
  }
}
