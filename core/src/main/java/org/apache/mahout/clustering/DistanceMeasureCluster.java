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
    super();
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.AbstractCluster#readFields(java.io.DataInput)
   */
  @Override
  public void readFields(DataInput in) throws IOException {
    String dm = in.readUTF();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      this.measure = (DistanceMeasure) ((Class<?>) ccl.loadClass(dm)).newInstance();
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
    super.readFields(in);
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.AbstractCluster#write(java.io.DataOutput)
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(measure.getClass().getName());
    super.write(out);
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.AbstractCluster#pdf(org.apache.mahout.math.VectorWritable)
   */
  @Override
  public double pdf(VectorWritable vw) {
    return Math.exp(-measure.distance(vw.get(), getCenter()));
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.dirichlet.models.Model#sampleFromPosterior()
   */
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
