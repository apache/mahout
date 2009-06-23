package org.apache.mahout.clustering.fuzzykmeans;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;


/**
 *
 *
 **/
public class FuzzyKMeansOutput implements Writable {
  //parallel arrays
  private SoftCluster[] clusters;
  private double[] probabilities;

  public FuzzyKMeansOutput() {
  }

  public FuzzyKMeansOutput(int size) {
    clusters = new SoftCluster[size];
    probabilities = new double[size];
  }

  public SoftCluster[] getClusters() {
    return clusters;
  }

  public double[] getProbabilities() {
    return probabilities;
  }

  public void add(int i, SoftCluster softCluster, double probWeight) {
    clusters[i] = softCluster;
    probabilities[i] = probWeight;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(clusters.length);
    for (int i = 0; i < clusters.length; i++) {
      clusters[i].write(out);
    }
    out.writeInt(probabilities.length);
    for (int i = 0; i < probabilities.length; i++) {
      out.writeDouble(probabilities[i]);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int numClusters = in.readInt();
    clusters = new SoftCluster[numClusters];
    for (int i = 0; i < numClusters; i++) {
      clusters[i] = new SoftCluster();
      clusters[i].readFields(in);
    }
    int numProbs = in.readInt();
    probabilities = new double[numProbs];
    for (int i = 0; i < numProbs; i++) {
      probabilities[i] = in.readDouble();
    }
  }
}
