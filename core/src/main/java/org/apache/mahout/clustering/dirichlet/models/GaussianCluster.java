package org.apache.mahout.clustering.dirichlet.models;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class GaussianCluster extends AbstractCluster {

  public GaussianCluster() {
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

  @Override
  public double pdf(VectorWritable vw) {
    Vector x = vw.get();
    // return the average of the component pdfs
    // TODO: is this reasonable? correct?
    double pdf = 0;
    for (int i = 0; i < x.size(); i++) {
      double x2 = x.get(i);
      double m = getCenter().get(i);
      double s = getRadius().get(i);
      double dNorm = UncommonDistributions.dNorm(x2, m, s);
      pdf += dNorm;
    }
    return pdf / x.size();
  }

}
