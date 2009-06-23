package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.Vector;

import java.io.DataOutput;
import java.io.IOException;
import java.io.DataInput;


/**
 *
 *
 **/
public class KMeansInfo implements Writable {

  private int points;
  private Vector pointTotal;

  public KMeansInfo() {
  }

  public KMeansInfo(int points, Vector pointTotal) {
    this.points = points;
    this.pointTotal = pointTotal;
  }

  public int getPoints() {
    return points;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(points);
    out.writeUTF(pointTotal.getClass().getSimpleName().toString());
    pointTotal.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.points = in.readInt();
    String className = in.readUTF();
    pointTotal = Cluster.vectorNameToVector(className);
    pointTotal.readFields(in);
  }
}
