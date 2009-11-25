/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.linalg;

import org.apache.mahout.matrix.matrix.DoubleMatrix1D;
import org.apache.mahout.matrix.matrix.DoubleMatrix2D;
/**
Sequential implementation of the Basic Linear Algebra System.

@author wolfgang.hoschek@cern.ch
@version 0.9, 16/04/2000 
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public class SeqBlas implements Blas {
  /**
  Little trick to allow for "aliasing", that is, renaming this class.
  Time and again writing code like
  <p>
  <tt>SeqBlas.blas.dgemm(...);</tt>
  <p>
  is a bit awkward. Using the aliasing you can instead write
  <p>
  <tt>Blas B = SeqBlas.blas; <br>
  B.dgemm(...);</tt>
  */
  public static final Blas seqBlas = new SeqBlas();
  
  private static final org.apache.mahout.jet.math.Functions F = org.apache.mahout.jet.math.Functions.functions;
/**
Makes this class non instantiable, but still let's others inherit from it.
*/
protected SeqBlas() {}
public void assign(DoubleMatrix2D A, org.apache.mahout.matrix.function.DoubleFunction function) {
  A.assign(function);
}
public void assign(DoubleMatrix2D A, DoubleMatrix2D B, org.apache.mahout.matrix.function.DoubleDoubleFunction function) {
  A.assign(B,function);
}
public double dasum(DoubleMatrix1D x) {
  return x.aggregate(F.plus, F.abs);
}
public void daxpy(double alpha, DoubleMatrix1D x, DoubleMatrix1D y) {
  y.assign(x,F.plusMult(alpha));
}
public void daxpy(double alpha, DoubleMatrix2D A, DoubleMatrix2D B) {
  B.assign(A, F.plusMult(alpha));
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
public void dgemm(boolean transposeA, boolean transposeB, double alpha, DoubleMatrix2D A, DoubleMatrix2D B, double beta, DoubleMatrix2D C) {
  A.zMult(B,C,alpha,beta,transposeA,transposeB);
}
public void dgemv(boolean transposeA, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta, DoubleMatrix1D y) {
  A.zMult(x,y,alpha,beta,transposeA);
}
public void dger(double alpha, DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A) {
  org.apache.mahout.jet.math.PlusMult fun = org.apache.mahout.jet.math.PlusMult.plusMult(0);
  for (int i=A.rows(); --i >= 0; ) {
    fun.multiplicator = alpha * x.getQuick(i);
     A.viewRow(i).assign(y,fun);
    
  }
}
public double dnrm2(DoubleMatrix1D x) {
  return Math.sqrt(Algebra.DEFAULT.norm2(x));
}
public void drot(DoubleMatrix1D x, DoubleMatrix1D y, double c, double s) {
  x.checkSize(y);
  DoubleMatrix1D tmp = x.copy();
  
  x.assign(F.mult(c));
  x.assign(y,F.plusMult(s));

  y.assign(F.mult(c));
  y.assign(tmp,F.minusMult(s));
}
public void drotg(double a, double b, double rotvec[]) {
  double c,s,roe,scale,r,z,ra,rb;

  roe = b;

  if (Math.abs(a) > Math.abs(b)) roe = a;

  scale = Math.abs(a) + Math.abs(b);

  if (scale != 0.0) {

    ra = a/scale;
    rb = b/scale;
    r = scale*Math.sqrt(ra*ra + rb*rb);
    r = sign(1.0,roe)*r;
    c = a/r;
    s = b/r;
    z = 1.0;
    if (Math.abs(a) > Math.abs(b)) z = s;
    if ((Math.abs(b) >= Math.abs(a)) && (c != 0.0)) z = 1.0/c;

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
  x.assign(F.mult(alpha));
}

public void dscal(double alpha, DoubleMatrix2D A) {
  A.assign(F.mult(alpha));
}

public void dswap(DoubleMatrix1D x, DoubleMatrix1D y) {
  y.swap(x);
}
public void dswap(DoubleMatrix2D A, DoubleMatrix2D B) {
  //B.swap(A); not yet implemented
  A.checkShape(B);
  for(int i = A.rows(); --i >= 0;) A.viewRow(i).swap(B.viewRow(i));
}
public void dsymv(boolean isUpperTriangular, double alpha, DoubleMatrix2D A, DoubleMatrix1D x, double beta, DoubleMatrix1D y) {
  if (isUpperTriangular) A = A.viewDice();
  Property.DEFAULT.checkSquare(A);
  int size = A.rows();
  if (size != x.size() || size!=y.size()) {
    throw new IllegalArgumentException(A.toStringShort() + ", " + x.toStringShort() + ", " + y.toStringShort());
  }
  DoubleMatrix1D tmp = x.like();
  for (int i = 0; i < size; i++) {
    double sum = 0;
    for (int j = 0; j <= i; j++) {
      sum += A.getQuick(i,j) * x.getQuick(j);
    }
    for (int j = i + 1; j < size; j++) {
      sum += A.getQuick(j,i) * x.getQuick(j);
    }
    tmp.setQuick(i, alpha * sum + beta * y.getQuick(i));
  }
  y.assign(tmp);
}
public void dtrmv(boolean isUpperTriangular, boolean transposeA, boolean isUnitTriangular, DoubleMatrix2D A, DoubleMatrix1D x) {
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
  }
  else {
    for (int i = 0; i < size; i++) {
      y.setQuick(i, A.getQuick(i,i));
    }
  }
  
  for (int i = 0; i < size; i++) {
    double sum = 0;
    if (!isUpperTriangular) {
      for (int j = 0; j < i; j++) {
        sum += A.getQuick(i,j) * x.getQuick(j);
      }
      sum += y.getQuick(i) * x.getQuick(i);
    }
    else {
      sum += y.getQuick(i) * x.getQuick(i);
      for (int j = i + 1; j < size; j++) {
        sum += A.getQuick(i,j) * x.getQuick(j);      }
    }
    b.setQuick(i,sum);
  }
  x.assign(b);
}
public int idamax(DoubleMatrix1D x) {
  int maxIndex = -1;
  double maxValue = Double.MIN_VALUE;
  for (int i=x.size(); --i >= 0; ) {
    double v = Math.abs(x.getQuick(i));
    if (v > maxValue) {
      maxValue = v;
      maxIndex = i;
    }
  }
  return maxIndex;
}
/**
Implements the FORTRAN sign (not sin) function.
See the code for details.
@param  a   a
@param  b   b
*/
private double sign(double a, double b) {
  if (b < 0.0) {
    return -Math.abs(a);
  } else {
    return Math.abs(a);      
  }
}
}
