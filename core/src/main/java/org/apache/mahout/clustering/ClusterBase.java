package org.apache.mahout.clustering;

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.hadoop.io.Writable;

import java.io.DataOutput;
import java.io.IOException;
import java.io.DataInput;


/**
 *
 *
 **/
public abstract class ClusterBase implements Writable {
  // this canopy's canopyId
  protected int id;
  // the current center
  protected Vector center = new SparseVector(0);

  // the number of points in the canopy
  protected int numPoints = 0;

    // the total of all points added to the cluster
  protected Vector pointTotal = null;

  public Vector getPointTotal() {
    return pointTotal;
  }

  public int getId() {
    return id;
  }

  /**
   * Return the center point
   *
   * @return the center of the Canopy
   */
  public Vector getCenter() {
    return center;
  }

  public int getNumPoints() {
    return numPoints;
  }

  public abstract String asFormatString();
  /**
   * Simply writes out the id, and that's it!
   * @param out The {@link java.io.DataOutput}
   * @throws IOException
   */
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
  }

  /**
   * Reads in the id, nothing else
   * @param in
   * @throws IOException
   */
  public void readFields(DataInput in) throws IOException {
    id = in.readInt();
  }
}
