package org.apache.mahout.math;

import java.util.Iterator;


public interface VectorIterable extends Iterable<MatrixSlice> {

  Iterator<MatrixSlice> iterateAll();

  int numSlices();

  int numRows();

  int numCols();

  /**
   * Return a new vector with cardinality equal to getNumRows() of this matrix which is the matrix product of the
   * recipient and the argument
   *
   * @param v a vector with cardinality equal to getNumCols() of the recipient
   * @return a new vector (typically a DenseVector)
   * @throws CardinalityException if this.getNumRows() != v.size()
   */
  Vector times(Vector v);

  /**
   * Convenience method for producing this.transpose().times(this.times(v)), which can be implemented with only one pass
   * over the matrix, without making the transpose() call (which can be expensive if the matrix is sparse)
   *
   * @param v a vector with cardinality equal to getNumCols() of the recipient
   * @return a new vector (typically a DenseVector) with cardinality equal to that of the argument.
   * @throws CardinalityException if this.getNumCols() != v.size()
   */
  Vector timesSquared(Vector v);

}
