package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import org.apache.mahout.math.Vector;

public class MutableElement implements Vector.Element {

  private int index;
  private double value;

  MutableElement(int index, double value) {
    this.index = index;
    this.value = value;
  }

  @Override
  public double get() {
    return value;
  }

  @Override
  public int index() {
    return index;
  }

  public void setIndex(int index) {
    this.index = index;
  }

  @Override
  public void set(double value) {
    this.value = value;
  }
}
