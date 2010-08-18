package org.apache.mahout.clustering.dirichlet.models;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class GaussianCluster extends AbstractCluster {

  public GaussianCluster() {
    super();
  }

  public GaussianCluster(Vector point, int id2) {
    super(point, id2);
  }

  public GaussianCluster(Vector center, Vector radius, int id) {
    super(center, radius, id);
  }

  @Override
  public String getIdentifier() {
    return "GC:" + getId();
  }

  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return new GaussianCluster(getCenter(), getRadius(), getId());
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.dirichlet.models.Model#pdf(java.lang.Object)
   */
  @Override
  public double pdf(VectorWritable vw) {
    Vector x = vw.get();
    // return the product of the component pdfs
    // TODO: is this reasonable? correct?
    double pdf = pdf(x, getRadius().get(0));
    for (int i = 1; i < x.size(); i++) {
      pdf *= pdf(x, getRadius().get(i));
    }
    return pdf;
  }

  /**
   * Calculate a pdf using the supplied sample and stdDev
   * 
   * @param x
   *          a Vector sample
   * @param sd
   *          a double std deviation
   */
  private double pdf(Vector x, double sd) {
    double sd2 = sd * sd;
    double exp = -(x.dot(x) - 2 * x.dot(getCenter()) + getCenter().dot(getCenter())) / (2 * sd2);
    double ex = Math.exp(exp);
    return ex / (sd * SQRT2PI);
  }

}
