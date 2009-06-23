package org.apache.mahout.clustering.fuzzykmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.clustering.kmeans.Cluster;

import java.io.DataOutput;
import java.io.IOException;
import java.io.DataInput;


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
    out.writeUTF(pointTotal.getClass().getSimpleName().toString());
    pointTotal.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.probability = in.readDouble();
    String className = in.readUTF();
    pointTotal = Cluster.vectorNameToVector(className);
    pointTotal.readFields(in);
  }
}