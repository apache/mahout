package org.apache.mahout.math;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: 8/9/11 Time: 10:36 PM To change this template use
 * File | Settings | File Templates.
 */
public class DiagonalMatrix extends AbstractMatrix {
  private Vector diagonal;

  public DiagonalMatrix(Vector values) {
    this.diagonal = values;
    super.cardinality[0] = values.size();
    super.cardinality[1] = values.size();
  }

  public DiagonalMatrix(Matrix values) {
    this(values.viewDiagonal());
  }

  public DiagonalMatrix(double value, int size) {
    this(new ConstantVector(value, size));
  }

  public DiagonalMatrix(double[] values) {
    this.diagonal = new DenseVector(values);
  }

  public static DiagonalMatrix identity(int size) {
    return new DiagonalMatrix(1, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    throw new UnsupportedOperationException("Can't assign a column to a diagonal matrix");
  }

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  @Override
  public Matrix assignRow(int row, Vector other) {
    throw new UnsupportedOperationException("Can't assign a row to a diagonal matrix");
  }

  /**
   * Return the column at the given index
   *
   * @param column an int column index
   * @return a Vector at the index
   * @throws IndexException if the index is out of bounds
   */
  @Override
  public Vector getColumn(int column) {
    return new MatrixVectorView(this, 0, column, 1, 0);
  }

  /**
   * Return the row at the given index
   *
   * @param row an int row index
   * @return a Vector at the index
   * @throws IndexException if the index is out of bounds
   */
  @Override
  public Vector getRow(int row) {
    return new MatrixVectorView(this, row, 0, 0, 1);
  }

  /**
   * Provides a view of the diagonal of a matrix.
   */
  @Override
  public Vector viewDiagonal() {
    return this.diagonal;
  }

  /**
   * Return the value at the given location, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int row, int column) {
    if (row == column) {
      return diagonal.get(row);
    } else {
      return 0;
    }
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    return new SparseRowMatrix(size());
  }

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified
   * size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  @Override
  public Matrix like(int rows, int columns) {
    return new SparseRowMatrix(new int[]{rows, columns});
  }

  @Override
  public void setQuick(int row, int column, double value) {
    if (row == column) {
      diagonal.set(row, value);
    } else {
      throw new UnsupportedOperationException("Can't set off-diagonal element");
    }
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  @Override
  public int[] getNumNondefaultElements() {
    throw new UnsupportedOperationException("Don't understand how to implement this");
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the
   *                              receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    return new MatrixView(this, offset, size);
  }
}
