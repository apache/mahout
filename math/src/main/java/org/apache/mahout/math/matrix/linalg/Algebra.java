/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.GenericPermuting;
import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.bitvector.QuickBitVector;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.jet.math.Functions;
import org.apache.mahout.math.list.ObjectArrayList;
import org.apache.mahout.math.matrix.DoubleFactory2D;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Algebra extends PersistentObject {

  /**
   * A default Algebra object; has {@link Property#DEFAULT} attached for tolerance. Allows ommiting to construct an
   * Algebra object time and again.
   *
   * Note that this Algebra object is immutable. Any attempt to assign a new Property object to it (via method
   * <tt>setProperty</tt>), or to alter the tolerance of its property object (via <tt>property().setTolerance(...)</tt>)
   * will throw an exception.
   */
  public static final Algebra DEFAULT;

  /**
   * A default Algebra object; has {@link Property#ZERO} attached for tolerance. Allows ommiting to construct an Algebra
   * object time and again.
   *
   * Note that this Algebra object is immutable. Any attempt to assign a new Property object to it (via method
   * <tt>setProperty</tt>), or to alter the tolerance of its property object (via <tt>property().setTolerance(...)</tt>)
   * will throw an exception.
   */
  private static final Algebra ZERO;

  /** The property object attached to this instance. */
  private Property property;

  static {
    // don't use new Algebra(Property.DEFAULT.tolerance()), because then property object would be mutable.
    DEFAULT = new Algebra();
    DEFAULT.property = Property.DEFAULT; // immutable property object

    ZERO = new Algebra();
    ZERO.property = Property.ZERO; // immutable property object
  }

  /** Constructs a new instance with an equality tolerance given by <tt>Property.DEFAULT.tolerance()</tt>. */
  public Algebra() {
    this(Property.DEFAULT.tolerance());
  }

  /**
   * Constructs a new instance with the given equality tolerance.
   *
   * @param tolerance the tolerance to be used for equality operations.
   */
  public Algebra(double tolerance) {
    setProperty(new Property(tolerance));
  }

  /** Constructs and returns the cholesky-decomposition of the given matrix. */
  private static CholeskyDecomposition chol(DoubleMatrix2D matrix) {
    return new CholeskyDecomposition(matrix);
  }

  /**
   * Returns a copy of the receiver. The attached property object is also copied. Hence, the property object of the copy
   * is mutable.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    return new Algebra(property.tolerance());
  }

  /** Returns the condition of matrix <tt>A</tt>, which is the ratio of largest to smallest singular value. */
  public static double cond(DoubleMatrix2D A) {
    return svd(A).cond();
  }

  /**
   * Returns the determinant of matrix <tt>A</tt>.
   *
   * @return the determinant.
   */
  public static double det(DoubleMatrix2D A) {
    return lu(A).det();
  }

  /** Constructs and returns the Eigenvalue-decomposition of the given matrix. */
  private static EigenvalueDecomposition eig(DoubleMatrix2D matrix) {
    return new EigenvalueDecomposition(matrix);
  }

  /** Returns sqrt(a^2 + b^2) without under/overflow. */
  protected static double hypot(double a, double b) {
    double r;
    if (Math.abs(a) > Math.abs(b)) {
      r = b / a;
      r = Math.abs(a) * Math.sqrt(1 + r * r);
    } else if (b != 0) {
      r = a / b;
      r = Math.abs(b) * Math.sqrt(1 + r * r);
    } else {
      r = 0.0;
    }
    return r;
  }

  /** Returns sqrt(a^2 + b^2) without under/overflow. */
  private static org.apache.mahout.math.function.DoubleDoubleFunction hypotFunction() {
    return new DoubleDoubleFunction() {
      @Override
      public double apply(double a, double b) {
        return hypot(a, b);
      }
    };
  }

  /**
   * Returns the inverse or pseudo-inverse of matrix <tt>A</tt>.
   *
   * @return a new independent matrix; inverse(matrix) if the matrix is square, pseudoinverse otherwise.
   */
  public DoubleMatrix2D inverse(DoubleMatrix2D A) {
    if (property.isSquare(A) && property.isDiagonal(A)) {
      DoubleMatrix2D inv = A.copy();
      boolean isNonSingular = Diagonal.inverse(inv);
      if (!isNonSingular) {
        throw new IllegalArgumentException("A is singular.");
      }
      return inv;
    }
    return solve(A, DoubleFactory2D.dense.identity(A.rows()));
  }

  /** Constructs and returns the LU-decomposition of the given matrix. */
  private static LUDecomposition lu(DoubleMatrix2D matrix) {
    return new LUDecomposition(matrix);
  }

  /**
   * Inner product of two vectors; <tt>Sum(x[i] * y[i])</tt>. Also known as dot product. <br> Equivalent to
   * <tt>x.zDotProduct(y)</tt>.
   *
   * @param x the first source vector.
   * @param y the second source matrix.
   * @return the inner product.
   * @throws IllegalArgumentException if <tt>x.size() != y.size()</tt>.
   */
  public static double mult(DoubleMatrix1D x, DoubleMatrix1D y) {
    return x.zDotProduct(y);
  }

  /**
   * Linear algebraic matrix-vector multiplication; <tt>z = A * y</tt>. <tt>z[i] = Sum(A[i,j] * y[j]), i=0..A.rows()-1,
   * j=0..y.size()-1</tt>.
   *
   * @param A the source matrix.
   * @param y the source vector.
   * @return <tt>z</tt>; a new vector with <tt>z.size()==A.rows()</tt>.
   * @throws IllegalArgumentException if <tt>A.columns() != y.size()</tt>.
   */
  public static DoubleMatrix1D mult(DoubleMatrix2D A, DoubleMatrix1D y) {
    return A.zMult(y, null);
  }

  /**
   * Linear algebraic matrix-matrix multiplication; <tt>C = A x B</tt>. <tt>C[i,j] = Sum(A[i,k] * B[k,j]),
   * k=0..n-1</tt>. <br> Matrix shapes: <tt>A(m x n), B(n x p), C(m x p)</tt>.
   *
   * @param A the first source matrix.
   * @param B the second source matrix.
   * @return <tt>C</tt>; a new matrix holding the results, with <tt>C.rows()=A.rows(), C.columns()==B.columns()</tt>.
   * @throws IllegalArgumentException if <tt>B.rows() != A.columns()</tt>.
   */
  public static DoubleMatrix2D mult(DoubleMatrix2D A, DoubleMatrix2D B) {
    return A.zMult(B, null);
  }

  /**
   * Outer product of two vectors; Sets <tt>A[i,j] = x[i] * y[j]</tt>.
   *
   * @param x the first source vector.
   * @param y the second source vector.
   * @param A the matrix to hold the results. Set this parameter to <tt>null</tt> to indicate that a new result matrix
   *          shall be constructed.
   * @return A (for convenience only).
   * @throws IllegalArgumentException if <tt>A.rows() != x.size() || A.columns() != y.size()</tt>.
   */
  public static DoubleMatrix2D multOuter(DoubleMatrix1D x, DoubleMatrix1D y, DoubleMatrix2D A) {
    int rows = x.size();
    int columns = y.size();
    if (A == null) {
      A = x.like2D(rows, columns);
    }
    if (A.rows() != rows || A.columns() != columns) {
      throw new IllegalArgumentException();
    }

    for (int row = rows; --row >= 0;) {
      A.viewRow(row).assign(y);
    }

    for (int column = columns; --column >= 0;) {
      A.viewColumn(column).assign(x, org.apache.mahout.math.jet.math.Functions.mult);
    }
    return A;
  }

  /** Returns the one-norm of vector <tt>x</tt>, which is <tt>Sum(abs(x[i]))</tt>. */
  public static double norm1(DoubleMatrix1D x) {
    if (x.size() == 0) {
      return 0;
    }
    return x.aggregate(Functions.plus, org.apache.mahout.math.jet.math.Functions.abs);
  }

  /** Returns the one-norm of matrix <tt>A</tt>, which is the maximum absolute column sum. */
  public double norm1(DoubleMatrix2D A) {
    double max = 0;
    for (int column = A.columns(); --column >= 0;) {
      max = Math.max(max, norm1(A.viewColumn(column)));
    }
    return max;
  }

  /** Returns the two-norm (aka <i>euclidean norm</i>) of vector <tt>x</tt>; equivalent to <tt>mult(x,x)</tt>. */
  public double norm2(DoubleMatrix1D x) {
    return mult(x, x);
  }

  /** Returns the two-norm of matrix <tt>A</tt>, which is the maximum singular value; obtained from SVD. */
  public static double norm2(DoubleMatrix2D A) {
    return svd(A).norm2();
  }

  /** Returns the Frobenius norm of matrix <tt>A</tt>, which is <tt>Sqrt(Sum(A[i,j]<sup>2</sup>))</tt>. */
  public static double normF(DoubleMatrix2D A) {
    if (A.size() == 0) {
      return 0;
    }
    return A.aggregate(hypotFunction(), org.apache.mahout.math.jet.math.Functions.identity);
  }

  /** Returns the infinity norm of vector <tt>x</tt>, which is <tt>Max(abs(x[i]))</tt>. */
  public static double normInfinity(DoubleMatrix1D x) {
    // fix for bug reported by T.J.Hunt@open.ac.uk
    if (x.size() == 0) {
      return 0;
    }
    return x.aggregate(Functions.max, org.apache.mahout.math.jet.math.Functions.abs);
//  if (x.size()==0) return 0;
//  return x.aggregate(Functions.plus,org.apache.mahout.math.jet.math.Functions.abs);
//  double max = 0;
//  for (int i = x.size(); --i >= 0; ) {
//    max = Math.max(max, x.getQuick(i));
//  }
//  return max;
  }

  /** Returns the infinity norm of matrix <tt>A</tt>, which is the maximum absolute row sum. */
  public double normInfinity(DoubleMatrix2D A) {
    double max = 0;
    for (int row = A.rows(); --row >= 0;) {
      //max = Math.max(max, normInfinity(A.viewRow(row)));
      max = Math.max(max, norm1(A.viewRow(row)));
    }
    return max;
  }

  /**
   * Modifies the given vector <tt>A</tt> such that it is permuted as specified; Useful for pivoting. Cell <tt>A[i]</tt>
   * will go into cell <tt>A[indexes[i]]</tt>. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[2], A[3]<--A[3], A[4]<--A[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[1], A[3]<--A[2], A[4]<--A[3].
   * </pre>
   *
   * @param A       the vector to permute.
   * @param indexes the permutation indexes, must satisfy <tt>indexes.length==A.size() && indexes[i] >= 0 && indexes[i]
   *                < A.size()</tt>;
   * @param work    the working storage, must satisfy <tt>work.length >= A.size()</tt>; set <tt>work==null</tt> if you
   *                don't care about performance.
   * @return the modified <tt>A</tt> (for convenience only).
   * @throws IndexOutOfBoundsException if <tt>indexes.length != A.size()</tt>.
   */
  public static DoubleMatrix1D permute(DoubleMatrix1D A, int[] indexes, double[] work) {
    // check validity
    int size = A.size();
    if (indexes.length != size) {
      throw new IndexOutOfBoundsException("invalid permutation");
    }

    /*
    int i=size;
    int a;
    while (--i >= 0 && (a=indexes[i])==i) if (a < 0 || a >= size) throw new IndexOutOfBoundsException("invalid permutation");
    if (i<0) return; // nothing to permute
    */

    if (work == null || size > work.length) {
      work = A.toArray();
    } else {
      A.toArray(work);
    }
    for (int i = size; --i >= 0;) {
      A.setQuick(i, work[indexes[i]]);
    }
    return A;
  }

  /**
   * Constructs and returns a new row and column permuted <i>selection view</i> of matrix <tt>A</tt>; equivalent to
   * {@link DoubleMatrix2D#viewSelection(int[],int[])}. The returned matrix is backed by this matrix, so changes in the
   * returned matrix are reflected in this matrix, and vice-versa. Use idioms like <tt>result = permute(...).copy()</tt>
   * to generate an independent sub matrix.
   *
   * @return the new permuted selection view.
   */
  public static DoubleMatrix2D permute(DoubleMatrix2D A, int[] rowIndexes, int[] columnIndexes) {
    return A.viewSelection(rowIndexes, columnIndexes);
  }

  /**
   * Modifies the given matrix <tt>A</tt> such that it's columns are permuted as specified; Useful for pivoting. Column
   * <tt>A[i]</tt> will go into column <tt>A[indexes[i]]</tt>. Equivalent to <tt>permuteRows(transpose(A), indexes,
   * work)</tt>.
   *
   * @param A       the matrix to permute.
   * @param indexes the permutation indexes, must satisfy <tt>indexes.length==A.columns() && indexes[i] >= 0 &&
   *                indexes[i] < A.columns()</tt>;
   * @param work    the working storage, must satisfy <tt>work.length >= A.columns()</tt>; set <tt>work==null</tt> if
   *                you don't care about performance.
   * @return the modified <tt>A</tt> (for convenience only).
   * @throws IndexOutOfBoundsException if <tt>indexes.length != A.columns()</tt>.
   */
  public DoubleMatrix2D permuteColumns(DoubleMatrix2D A, int[] indexes, int[] work) {
    return permuteRows(A.viewDice(), indexes, work);
  }

  /**
   * Modifies the given matrix <tt>A</tt> such that it's rows are permuted as specified; Useful for pivoting. Row
   * <tt>A[i]</tt> will go into row <tt>A[indexes[i]]</tt>. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[2], A[3]<--A[3], A[4]<--A[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words A[0]<--A[0], A[1]<--A[4], A[2]<--A[1], A[3]<--A[2], A[4]<--A[3].
   * </pre>
   *
   * @param A       the matrix to permute.
   * @param indexes the permutation indexes, must satisfy <tt>indexes.length==A.rows() && indexes[i] >= 0 && indexes[i]
   *                < A.rows()</tt>;
   * @param work    the working storage, must satisfy <tt>work.length >= A.rows()</tt>; set <tt>work==null</tt> if you
   *                don't care about performance.
   * @return the modified <tt>A</tt> (for convenience only).
   * @throws IndexOutOfBoundsException if <tt>indexes.length != A.rows()</tt>.
   */
  public DoubleMatrix2D permuteRows(final DoubleMatrix2D A, int[] indexes, int[] work) {
    // check validity
    int size = A.rows();
    if (indexes.length != size) {
      throw new IndexOutOfBoundsException("invalid permutation");
    }

    /*
    int i=size;
    int a;
    while (--i >= 0 && (a=indexes[i])==i) if (a < 0 || a >= size) throw new IndexOutOfBoundsException("invalid permutation");
    if (i<0) return; // nothing to permute
    */

    int columns = A.columns();
    if (columns < size / 10) { // quicker
      double[] doubleWork = new double[size];
      for (int j = A.columns(); --j >= 0;) {
        permute(A.viewColumn(j), indexes, doubleWork);
      }
      return A;
    }

    Swapper swapper = new Swapper() {
      @Override
      public void swap(int a, int b) {
        A.viewRow(a).swap(A.viewRow(b));
      }
    };

    GenericPermuting.permute(indexes, swapper, work, null);
    return A;
  }

  /**
   * Linear algebraic matrix power; <tt>B = A<sup>k</sup> <==> B = A*A*...*A</tt>. <ul> <li><tt>p &gt;= 1: B =
   * A*A*...*A</tt>.</li> <li><tt>p == 0: B = identity matrix</tt>.</li> <li><tt>p &lt;  0: B =
   * pow(inverse(A),-p)</tt>.</li> </ul> Implementation: Based on logarithms of 2, memory usage minimized.
   *
   * @param A the source matrix; must be square; stays unaffected by this operation.
   * @param p the exponent, can be any number.
   * @return <tt>B</tt>, a newly constructed result matrix; storage-independent of <tt>A</tt>.
   * @throws IllegalArgumentException if <tt>!property().isSquare(A)</tt>.
   */
  public DoubleMatrix2D pow(DoubleMatrix2D A, int p) {
    // matrix multiplication based on log2 method: A*A*....*A is slow, ((A * A)^2)^2 * ... is faster
    // allocates two auxiliary matrices as work space

    Blas blas = SmpBlas.getSmpBlas(); // for parallel matrix mult; if not initialized defaults to sequential blas
    Property.DEFAULT.checkSquare(A);
    if (p < 0) {
      A = inverse(A);
      p = -p;
    }
    if (p == 0) {
      return DoubleFactory2D.dense.identity(A.rows());
    }
    DoubleMatrix2D T = A.like(); // temporary
    if (p == 1) {
      return T.assign(A);
    }  // safes one auxiliary matrix allocation
    if (p == 2) {
      blas.dgemm(false, false, 1, A, A, 0, T); // mult(A,A); // safes one auxiliary matrix allocation
      return T;
    }

    int k = QuickBitVector.mostSignificantBit(p); // index of highest bit in state "true"

    /*
    this is the naive version:
    DoubleMatrix2D B = A.copy();
    for (int i=0; i<p-1; i++) {
      B = mult(B,A);
    }
    return B;
    */

    // here comes the optimized version:
    //org.apache.mahout.math.Timer timer = new Timer().start();

    int i = 0;
    while (i <= k && (p & (1 << i)) == 0) { // while (bit i of p == false)
      // A = mult(A,A); would allocate a lot of temporary memory
      blas.dgemm(false, false, 1, A, A, 0, T); // A.zMult(A,T);
      DoubleMatrix2D swap = A;
      A = T;
      T = swap; // swap A with T
      i++;
    }

    DoubleMatrix2D B = A.copy();
    i++;
    for (; i <= k; i++) {
      // A = mult(A,A); would allocate a lot of temporary memory
      blas.dgemm(false, false, 1, A, A, 0, T); // A.zMult(A,T);
      DoubleMatrix2D swap = A;
      A = T;
      T = swap; // swap A with T

      if ((p & (1 << i)) != 0) { // if (bit i of p == true)
        // B = mult(B,A); would allocate a lot of temporary memory
        blas.dgemm(false, false, 1, B, A, 0, T); // B.zMult(A,T);
        swap = B;
        B = T;
        T = swap; // swap B with T
      }
    }
    //timer.stop().display();
    return B;
  }

  /**
   * Returns the property object attached to this Algebra, defining tolerance.
   *
   * @return the Property object.
   * @see #setProperty(Property)
   */
  public Property property() {
    return property;
  }

  /** Constructs and returns the QR-decomposition of the given matrix. */
  private static QRDecomposition qr(DoubleMatrix2D matrix) {
    return new QRDecomposition(matrix);
  }

  /** Returns the effective numerical rank of matrix <tt>A</tt>, obtained from Singular Value Decomposition. */
  public static int rank(DoubleMatrix2D A) {
    return svd(A).rank();
  }

  /**
   * Attaches the given property object to this Algebra, defining tolerance.
   *
   * @param property the Property object to be attached.
   * @throws UnsupportedOperationException if <tt>this==DEFAULT && property!=this.property()</tt> - The DEFAULT Algebra
   *                                       object is immutable.
   * @throws UnsupportedOperationException if <tt>this==ZERO && property!=this.property()</tt> - The ZERO Algebra object
   *                                       is immutable.
   * @see #property
   */
  public void setProperty(Property property) {
    if (this == DEFAULT && property != this.property) {
      throw new IllegalArgumentException("Attempted to modify immutable object.");
    }
    if (this == ZERO && property != this.property) {
      throw new IllegalArgumentException("Attempted to modify immutable object.");
    }
    this.property = property;
  }

  /**
   * Solves A*X = B.
   *
   * @return X; a new independent matrix; solution if A is square, least squares solution otherwise.
   */
  public static DoubleMatrix2D solve(DoubleMatrix2D A, DoubleMatrix2D B) {
    return (A.rows() == A.columns() ? (lu(A).solve(B)) : (qr(A).solve(B)));
  }

  /**
   * Solves X*A = B, which is also A'*X' = B'.
   *
   * @return X; a new independent matrix; solution if A is square, least squares solution otherwise.
   */
  public DoubleMatrix2D solveTranspose(DoubleMatrix2D A, DoubleMatrix2D B) {
    return solve(transpose(A), transpose(B));
  }

  /**
   * Copies the columns of the indicated rows into a new sub matrix. <tt>sub[0..rowIndexes.length-1,0..columnTo-columnFrom]
   * = A[rowIndexes(:),columnFrom..columnTo]</tt>; The returned matrix is <i>not backed</i> by this matrix, so changes
   * in the returned matrix are <i>not reflected</i> in this matrix, and vice-versa.
   *
   * @param A          the source matrix to copy from.
   * @param rowIndexes the indexes of the rows to copy. May be unsorted.
   * @param columnFrom the index of the first column to copy (inclusive).
   * @param columnTo   the index of the last column to copy (inclusive).
   * @return a new sub matrix; with <tt>sub.rows()==rowIndexes.length; sub.columns()==columnTo-columnFrom+1</tt>.
   * @throws IndexOutOfBoundsException if <tt>columnFrom<0 || columnTo-columnFrom+1<0 || columnTo+1>matrix.columns() ||
   *                                   for any row=rowIndexes[i]: row < 0 || row >= matrix.rows()</tt>.
   */
  private static DoubleMatrix2D subMatrix(DoubleMatrix2D A, int[] rowIndexes, int columnFrom, int columnTo) {
    int width = columnTo - columnFrom + 1;
    int rows = A.rows();
    A = A.viewPart(0, columnFrom, rows, width);
    DoubleMatrix2D sub = A.like(rowIndexes.length, width);

    for (int r = rowIndexes.length; --r >= 0;) {
      int row = rowIndexes[r];
      if (row < 0 || row >= rows) {
        throw new IndexOutOfBoundsException("Illegal Index");
      }
      sub.viewRow(r).assign(A.viewRow(row));
    }
    return sub;
  }

  /**
   * Copies the rows of the indicated columns into a new sub matrix. <tt>sub[0..rowTo-rowFrom,0..columnIndexes.length-1]
   * = A[rowFrom..rowTo,columnIndexes(:)]</tt>; The returned matrix is <i>not backed</i> by this matrix, so changes in
   * the returned matrix are <i>not reflected</i> in this matrix, and vice-versa.
   *
   * @param A             the source matrix to copy from.
   * @param rowFrom       the index of the first row to copy (inclusive).
   * @param rowTo         the index of the last row to copy (inclusive).
   * @param columnIndexes the indexes of the columns to copy. May be unsorted.
   * @return a new sub matrix; with <tt>sub.rows()==rowTo-rowFrom+1; sub.columns()==columnIndexes.length</tt>.
   * @throws IndexOutOfBoundsException if <tt>rowFrom<0 || rowTo-rowFrom+1<0 || rowTo+1>matrix.rows() || for any
   *                                   col=columnIndexes[i]: col < 0 || col >= matrix.columns()</tt>.
   */
  private static DoubleMatrix2D subMatrix(DoubleMatrix2D A, int rowFrom, int rowTo, int[] columnIndexes) {
    if (rowTo - rowFrom >= A.rows()) {
      throw new IndexOutOfBoundsException("Too many rows");
    }
    int height = rowTo - rowFrom + 1;
    int columns = A.columns();
    A = A.viewPart(rowFrom, 0, height, columns);
    DoubleMatrix2D sub = A.like(height, columnIndexes.length);

    for (int c = columnIndexes.length; --c >= 0;) {
      int column = columnIndexes[c];
      if (column < 0 || column >= columns) {
        throw new IndexOutOfBoundsException("Illegal Index");
      }
      sub.viewColumn(c).assign(A.viewColumn(column));
    }
    return sub;
  }

  /**
   * Constructs and returns a new <i>sub-range view</i> which is the sub matrix <tt>A[fromRow..toRow,fromColumn..toColumn]</tt>.
   * The returned matrix is backed by this matrix, so changes in the returned matrix are reflected in this matrix, and
   * vice-versa. Use idioms like <tt>result = subMatrix(...).copy()</tt> to generate an independent sub matrix.
   *
   * @param A          the source matrix.
   * @param fromRow    The index of the first row (inclusive).
   * @param toRow      The index of the last row (inclusive).
   * @param fromColumn The index of the first column (inclusive).
   * @param toColumn   The index of the last column (inclusive).
   * @return a new sub-range view.
   * @throws IndexOutOfBoundsException if <tt>fromColumn<0 || toColumn-fromColumn+1<0 || toColumn>=A.columns() ||
   *                                   fromRow<0 || toRow-fromRow+1<0 || toRow>=A.rows()</tt>
   */
  public static DoubleMatrix2D subMatrix(DoubleMatrix2D A, int fromRow, int toRow, int fromColumn, int toColumn) {
    return A.viewPart(fromRow, fromColumn, toRow - fromRow + 1, toColumn - fromColumn + 1);
  }

  /** Constructs and returns the SingularValue-decomposition of the given matrix. */
  private static SingularValueDecomposition svd(DoubleMatrix2D matrix) {
    return new SingularValueDecomposition(matrix);
  }

  /**
   * Returns a String with (propertyName, propertyValue) pairs. Useful for debugging or to quickly get the rough
   * picture. For example,
   * <pre>
   * cond          : 14.073264490042144
   * det           : Illegal operation or error: Matrix must be square.
   * norm1         : 0.9620244354009628
   * norm2         : 3.0
   * normF         : 1.304841791648992
   * normInfinity  : 1.5406551198102534
   * rank          : 3
   * trace         : 0
   * </pre>
   */
  public String toString(DoubleMatrix2D matrix) {
    final ObjectArrayList<String> names = new ObjectArrayList<String>();
    final ObjectArrayList<String> values = new ObjectArrayList<String>();

    // determine properties
    names.add("cond");
    String unknown = "Illegal operation or error: ";
    try {
      values.add(String.valueOf(cond(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("det");
    try {
      values.add(String.valueOf(det(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("norm1");
    try {
      values.add(String.valueOf(norm1(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("norm2");
    try {
      values.add(String.valueOf(norm2(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("normF");
    try {
      values.add(String.valueOf(normF(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("normInfinity");
    try {
      values.add(String.valueOf(normInfinity(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("rank");
    try {
      values.add(String.valueOf(rank(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }

    names.add("trace");
    try {
      values.add(String.valueOf(trace(matrix)));
    }
    catch (IllegalArgumentException exc) {
      values.add(unknown + exc.getMessage());
    }


    // sort ascending by property name
    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        return Property.get(names, a).compareTo(Property.get(names, b));
      }
    };
    Swapper swapper = new Swapper() {
      @Override
      public void swap(int a, int b) {
        String tmp = names.get(a);
        names.set(a, names.get(b));
        names.set(b, tmp);
        tmp = values.get(a);
        values.set(a, values.get(b));
        values.set(b, tmp);
      }
    };
    Sorting.quickSort(0, names.size(), comp, swapper);

    // determine padding for nice formatting
    int maxLength = 0;
    for (int i = 0; i < names.size(); i++) {
      int length = (names.get(i)).length();
      maxLength = Math.max(length, maxLength);
    }

    // finally, format properties
    StringBuilder buf = new StringBuilder();
    for (int i = 0; i < names.size(); i++) {
      String name = ((String) names.get(i));
      buf.append(name);
      buf.append(Property.blanks(maxLength - name.length()));
      buf.append(" : ");
      buf.append(values.get(i));
      if (i < names.size() - 1) {
        buf.append('\n');
      }
    }

    return buf.toString();
  }

  /**
   * Returns the results of <tt>toString(A)</tt> and additionally the results of all sorts of decompositions applied to
   * the given matrix. Useful for debugging or to quickly get the rough picture. For example,
   * <pre>
   * A = 3 x 3 matrix
   * 249  66  68
   * 104 214 108
   * 144 146 293
   *
   * cond         : 3.931600417472078
   * det          : 9638870.0
   * norm1        : 497.0
   * norm2        : 473.34508217011404
   * normF        : 516.873292016525
   * normInfinity : 583.0
   * rank         : 3
   * trace        : 756.0
   *
   * density                      : 1.0
   * isDiagonal                   : false
   * isDiagonallyDominantByColumn : true
   * isDiagonallyDominantByRow    : true
   * isIdentity                   : false
   * isLowerBidiagonal            : false
   * isLowerTriangular            : false
   * isNonNegative                : true
   * isOrthogonal                 : false
   * isPositive                   : true
   * isSingular                   : false
   * isSkewSymmetric              : false
   * isSquare                     : true
   * isStrictlyLowerTriangular    : false
   * isStrictlyTriangular         : false
   * isStrictlyUpperTriangular    : false
   * isSymmetric                  : false
   * isTriangular                 : false
   * isTridiagonal                : false
   * isUnitTriangular             : false
   * isUpperBidiagonal            : false
   * isUpperTriangular            : false
   * isZero                       : false
   * lowerBandwidth               : 2
   * semiBandwidth                : 3
   * upperBandwidth               : 2
   *
   * -----------------------------------------------------------------------------
   * LUDecompositionQuick(A) --> isNonSingular(A), det(A), pivot, L, U, inverse(A)
   * -----------------------------------------------------------------------------
   * isNonSingular = true
   * det = 9638870.0
   * pivot = [0, 1, 2]
   *
   * L = 3 x 3 matrix
   * 1        0       0
   * 0.417671 1       0
   * 0.578313 0.57839 1
   *
   * U = 3 x 3 matrix
   * 249  66         68
   * 0 186.433735  79.598394
   * 0   0        207.635819
   *
   * inverse(A) = 3 x 3 matrix
   * 0.004869 -0.000976 -0.00077
   * -0.001548  0.006553 -0.002056
   * -0.001622 -0.002786  0.004816
   *
   * -----------------------------------------------------------------
   * QRDecomposition(A) --> hasFullRank(A), H, Q, R, pseudo inverse(A)
   * -----------------------------------------------------------------
   * hasFullRank = true
   *
   * H = 3 x 3 matrix
   * 1.814086 0        0
   * 0.34002  1.903675 0
   * 0.470797 0.428218 2
   *
   * Q = 3 x 3 matrix
   * -0.814086  0.508871  0.279845
   * -0.34002  -0.808296  0.48067
   * -0.470797 -0.296154 -0.831049
   *
   * R = 3 x 3 matrix
   * -305.864349 -195.230337 -230.023539
   * 0        -182.628353  467.703164
   * 0           0        -309.13388
   *
   * pseudo inverse(A) = 3 x 3 matrix
   * 0.006601  0.001998 -0.005912
   * -0.005105  0.000444  0.008506
   * -0.000905 -0.001555  0.002688
   *
   * --------------------------------------------------------------------------
   * CholeskyDecomposition(A) --> isSymmetricPositiveDefinite(A), L, inverse(A)
   * --------------------------------------------------------------------------
   * isSymmetricPositiveDefinite = false
   *
   * L = 3 x 3 matrix
   * 15.779734  0         0
   * 6.590732 13.059948  0
   * 9.125629  6.573948 12.903724
   *
   * inverse(A) = Illegal operation or error: Matrix is not symmetric positive definite.
   *
   * ---------------------------------------------------------------------
   * EigenvalueDecomposition(A) --> D, V, realEigenvalues, imagEigenvalues
   * ---------------------------------------------------------------------
   * realEigenvalues = 1 x 3 matrix
   * 462.796507 172.382058 120.821435
   * imagEigenvalues = 1 x 3 matrix
   * 0 0 0
   *
   * D = 3 x 3 matrix
   * 462.796507   0          0
   * 0        172.382058   0
   * 0          0        120.821435
   *
   * V = 3 x 3 matrix
   * -0.398877 -0.778282  0.094294
   * -0.500327  0.217793 -0.806319
   * -0.768485  0.66553   0.604862
   *
   * ---------------------------------------------------------------------
   * SingularValueDecomposition(A) --> cond(A), rank(A), norm2(A), U, S, V
   * ---------------------------------------------------------------------
   * cond = 3.931600417472078
   * rank = 3
   * norm2 = 473.34508217011404
   *
   * U = 3 x 3 matrix
   * 0.46657  -0.877519  0.110777
   * 0.50486   0.161382 -0.847982
   * 0.726243  0.45157   0.51832
   *
   * S = 3 x 3 matrix
   * 473.345082   0          0
   * 0        169.137441   0
   * 0          0        120.395013
   *
   * V = 3 x 3 matrix
   * 0.577296 -0.808174  0.116546
   * 0.517308  0.251562 -0.817991
   * 0.631761  0.532513  0.563301
   * </pre>
   */
  public String toVerboseString(DoubleMatrix2D matrix) {
/*
  StringBuffer buf = new StringBuffer();
  String unknown = "Illegal operation or error: ";
  String constructionException = "Illegal operation or error upon construction: ";

  buf.append("------------------------------------------------------------------\n");
  buf.append("LUDecomposition(A) --> isNonSingular, det, pivot, L, U, inverse(A)\n");
  buf.append("------------------------------------------------------------------\n");
*/

    StringBuilder buf = new StringBuilder();

    buf.append("A = ");
    buf.append(matrix);

    buf.append("\n\n").append(toString(matrix));
    buf.append("\n\n").append(Property.DEFAULT.toString(matrix));

    LUDecomposition lu = null;
    String constructionException = "Illegal operation or error upon construction of ";
    try {
      lu = new LUDecomposition(matrix);
    }
    catch (IllegalArgumentException exc) {
      buf.append("\n\n").append(constructionException).append(" LUDecomposition: ").append(exc.getMessage());
    }
    if (lu != null) {
      buf.append("\n\n").append(lu.toString());
    }

    QRDecomposition qr = null;
    try {
      qr = new QRDecomposition(matrix);
    }
    catch (IllegalArgumentException exc) {
      buf.append("\n\n").append(constructionException).append(" QRDecomposition: ").append(exc.getMessage());
    }
    if (qr != null) {
      buf.append("\n\n").append(qr.toString());
    }

    CholeskyDecomposition chol = null;
    try {
      chol = new CholeskyDecomposition(matrix);
    }
    catch (IllegalArgumentException exc) {
      buf.append("\n\n").append(constructionException).append(" CholeskyDecomposition: ").append(exc.getMessage());
    }
    if (chol != null) {
      buf.append("\n\n").append(chol.toString());
    }

    EigenvalueDecomposition eig = null;
    try {
      eig = new EigenvalueDecomposition(matrix);
    }
    catch (IllegalArgumentException exc) {
      buf.append("\n\n").append(constructionException).append(" EigenvalueDecomposition: ").append(exc.getMessage());
    }
    if (eig != null) {
      buf.append("\n\n").append(eig.toString());
    }

    SingularValueDecomposition svd = null;
    try {
      svd = new SingularValueDecomposition(matrix);
    }
    catch (IllegalArgumentException exc) {
      buf.append("\n\n").append(constructionException).append(" SingularValueDecomposition: ").append(exc.getMessage());
    }
    if (svd != null) {
      buf.append("\n\n").append(svd.toString());
    }

    return buf.toString();
  }

  /** Returns the sum of the diagonal elements of matrix <tt>A</tt>; <tt>Sum(A[i,i])</tt>. */
  public static double trace(DoubleMatrix2D A) {
    double sum = 0;
    for (int i = Math.min(A.rows(), A.columns()); --i >= 0;) {
      sum += A.getQuick(i, i);
    }
    return sum;
  }

  /**
   * Constructs and returns a new view which is the transposition of the given matrix <tt>A</tt>. Equivalent to {@link
   * DoubleMatrix2D#viewDice A.viewDice()}. This is a zero-copy transposition, taking O(1), i.e. constant time. The
   * returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. Use idioms like <tt>result = transpose(A).copy()</tt> to generate an independent matrix. <p>
   * <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td>
   * <td>transpose ==></td> <td valign="top">3 x 2 matrix:<br> 1, 4 <br> 2, 5 <br> 3, 6</td> <td>transpose ==></td> <td
   * valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new transposed view.
   */
  public static DoubleMatrix2D transpose(DoubleMatrix2D A) {
    return A.viewDice();
  }

  /**
   * Modifies the matrix to be a lower trapezoidal matrix.
   *
   * @return <tt>A</tt> (for convenience only).
   */
  protected static DoubleMatrix2D trapezoidalLower(DoubleMatrix2D A) {
    int rows = A.rows();
    int columns = A.columns();
    for (int r = rows; --r >= 0;) {
      for (int c = columns; --c >= 0;) {
        if (r < c) {
          A.setQuick(r, c, 0);
        }
      }
    }
    return A;
  }

}
