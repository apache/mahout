package org.apache.mahout.math.hadoop.decomposer;

import org.apache.mahout.math.DenseVector;

/**
 * TODO this is a horrible hack.  Make a proper writable subclass also.
 */
public class EigenVector extends DenseVector {

  public EigenVector(DenseVector v, double eigenValue, double cosAngleError, int order) {
    super(v, false);
    setName("e|" + order +"| = |"+eigenValue+"|, err = "+cosAngleError);
  }

  public double getEigenValue() {
    return parseMetaData()[1];
  }

  public double getCosAngleError() {
    return parseMetaData()[2];
  }

  public int getIndex() {
    return (int)parseMetaData()[0];
  }

  protected double[] parseMetaData() {
    double[] m = new double[3];
    String[] s = getName().split(" = ");
    m[0] = Double.parseDouble(s[0].split("|")[1]);
    m[1] = Double.parseDouble(s[1].split("|")[1]);
    m[2] = Double.parseDouble(s[2].substring(1));
    return m;
  }

}
