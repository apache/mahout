package org.apache.mahout.clustering.fuzzykmeans;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;


/**
 *
 *
 **/
public class FuzzyKMeansInfo implements Writable {

  private double probability;
  private Vector pointTotal;

  public int combinerPass = 0;

  public FuzzyKMeansInfo() {
  }

  public FuzzyKMeansInfo(double probability, Vector pointTotal) {
    this.probability = probability;
    this.pointTotal = pointTotal;
  }

  public FuzzyKMeansInfo(double probability, Vector pointTotal, int combinerPass) {
    this.probability = probability;
    this.pointTotal = pointTotal;
    this.combinerPass = combinerPass;
  }

  public Vector getVector() {
    return pointTotal;
  }

  public double getProbability() {
    return probability;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(probability);
    AbstractVector.writeVector(out, pointTotal);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.probability = in.readDouble();
    this.pointTotal = AbstractVector.readVector(in);
  }
}