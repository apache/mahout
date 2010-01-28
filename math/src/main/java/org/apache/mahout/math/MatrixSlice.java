package org.apache.mahout.math;


public class MatrixSlice {
  private Vector v;
  private int index;
  public MatrixSlice(Vector v, int index) {
    this.v = v;
    this.index = index;
  }

  public Vector vector() { return v; }
  public int index() { return index; }
}

