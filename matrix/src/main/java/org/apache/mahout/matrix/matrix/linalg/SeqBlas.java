/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.linalg;

import org.apache.mahout.jet.math.Functions;
import org.apache.mahout.jet.math.PlusMult;
import org.apache.mahout.matrix.function.DoubleDoubleFunction;
import org.apache.mahout.matrix.matrix.DoubleMatrix1D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
 Sequential implementation of the Basic Linear Algebra System.

 @author wolfgang.hoschek@cern.ch
 @version 0.9, 16/04/2000
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class SeqBlas implements Blas {

  public static final Blas seqBlas = new SeqBlas();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private SeqBlas() {
  }

  @Override
  public void assign(DoubleMatrix2D A, org.apache.mahout.matrix.function.DoubleFunction function) {
    A.assign(function);
  }

  @Override
  public void assign(DoubleMatrix2D A, DoubleMatrix2D B,
                     DoubleDoubleFunction function) {
    A.assign(B, function);
  }

  @Override
  public double dasum(DoubleMatrix1D x) {
    return x.aggregate(Functions.plus, Functions.abs);
  }

  @Override
  public void daxpy(double alpha, DoubleMatrix1D x, DoubleMatrix1D y) {
    y.assign(x, Functions.plusMult(alpha));
  }

  @Override
  public void daxpy(double alpha, DoubleMatrix2D A, DoubleMatrix2D B) {
    B.assign(A, Functions.plusMult(alpha));
  }

  @Override
  public void dcopy(DoubleMatrix1D x, DoubleMatrix1D y) {
    y.assign(x);
  }

  @Override
  public void dcopy(DoubleMatrix2D A, DoubleMatrix2D B) {
    B.assign(A);
  }

  @Override
  public double ddot(DoubleMatrix1D x, DoubleMatrix1D y) {
    return x.zDotProduct(y);
  }

  @Override
  public void dgemm(boolean transposeA, boolean transposeB, double alpha, DoubleMatrix2D A, DoubleMatrix2D B,
                    double beta, DoubleMatrix2D C) {
    A.zMult(B, C, alpha, beta, transposeA, transposeB);
  }

  @Override
  public void dgemv(boolean transposeA, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta,
                    DoubleMatrix1D y) {
    A.zMult(x, y, alpha, beta, transposeA);
  }

  @Override
  public void dger(double alpha, DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A) {
    PlusMult fun = org.apache.mahout.jet.math.PlusMult.plusMult(0);
    for (int i = A.rows(); --i >= 0;) {
      fun.setMultiplicator(alpha * x.getQuick(i));
      A.viewRow(i).assign(y, fun);

    }
  }

  @Override
  public double dnrm2(DoubleMatrix1D x) {
    return Math.sqrt(Algebra.DEFAULT.norm2(x));
  }

  @Override
  public void drot(DoubleMatrix1D x, DoubleMatrix1D y, double c, double s) {
    x.checkSize(y);
    DoubleMatrix1D tmp = x.copy();

    x.assign(Functions.mult(c));
    x.assign(y, Functions.plusMult(s));

    y.assign(Functions.mult(c));
    y.assign(tmp, Functions.minusMult(s));
  }

  @Override
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

  @Override
  public void dscal(double alpha, DoubleMatrix1D x) {
    x.assign(Functions.mult(alpha));
  }

  @Override
  public void dscal(double alpha, DoubleMatrix2D A) {
    A.assign(Functions.mult(alpha));
  }

  @Override
  public void dswap(DoubleMatrix1D x, DoubleMatrix1D y) {
    y.swap(x);
  }

  @Override
  public void dswap(DoubleMatrix2D A, DoubleMatrix2D B) {
    //B.swap(A); not yet implemented
    A.checkShape(B);
    for (int i = A.rows(); --i >= 0;) {
      A.viewRow(i).swap(B.viewRow(i));
    }
  }

  @Override
  public void dsymv(boolean isUpperTriangular, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta,
                    DoubleMatrix1D y) {
    if (isUpperTriangular) {
      A = A.viewDice();
    }
    Property.DEFAULT.checkSquare(A);
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

  @Override
  public void dtrmv(boolean isUpperTriangular, boolean transposeA, boolean isUnitTriangular, DoubleMatrix2D A,
                    DoubleMatrix1D x) {
    if (transposeA) {
      A = A.viewDice();
      isUpperTriangular = !isUpperTriangular;
    }

    Property.DEFAULT.checkSquare(A);
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

  @Override
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
    if (b < 0.0) {
      return -Math.abs(a);
    } else {
      return Math.abs(a);
    }
  }
}
