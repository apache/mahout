package org.apache.mahout.matrix;

public class SquareRootFunction implements UnaryFunction {

  @Override
  public double apply(double arg1) {
    return Math.abs(arg1);
  }

}
