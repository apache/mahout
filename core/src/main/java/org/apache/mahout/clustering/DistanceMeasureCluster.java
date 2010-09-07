package org.apache.mahout.clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class DistanceMeasureCluster extends AbstractCluster {

  protected DistanceMeasure measure;

  public DistanceMeasureCluster(Vector point, int id, DistanceMeasure measure) {
    super(point, id);
    this.measure = measure;
  }

  public DistanceMeasureCluster() {
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    String dm = in.readUTF();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      this.measure = ccl.loadClass(dm).asSubclass(DistanceMeasure.class).newInstance();
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
    super.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(measure.getClass().getName());
    super.write(out);
  }

  @Override
  public double pdf(VectorWritable vw) {
    return Math.exp(-measure.distance(vw.get(), getCenter()));
  }

  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return new DistanceMeasureCluster(getCenter(), getId(), measure);
  }

  public DistanceMeasure getMeasure() {
    return measure;
  }

  /**
   * @param measure the measure to set
   */
  public void setMeasure(DistanceMeasure measure) {
    this.measure = measure;
  }

  @Override
  public String getIdentifier() {
    return "DMC:" + getId();
  }

}
