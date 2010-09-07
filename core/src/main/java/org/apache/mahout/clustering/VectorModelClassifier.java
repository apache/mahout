package org.apache.mahout.clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

public class VectorModelClassifier extends AbstractVectorClassifier {

  private final List<Model<VectorWritable>> models;

  public VectorModelClassifier(List<Model<VectorWritable>> models) {
    this.models = models;
  }

  @Override
  public Vector classify(Vector instance) {
    Vector pdfs = new DenseVector(models.size());
    if (models.get(0) instanceof SoftCluster) {
      List<SoftCluster> clusters = new ArrayList<SoftCluster>();
      List<Double> distances = new ArrayList<Double>();
      for (Model<VectorWritable> model : models) {
        SoftCluster sc = (SoftCluster) model;
        clusters.add(sc);
        distances.add(sc.getMeasure().distance(instance, sc.getCenter()));
      }
      return new FuzzyKMeansClusterer().computePi(clusters, distances);
    } else {
      int i = 0;
      for (Model<VectorWritable> model : models) {
        pdfs.set(i++, model.pdf(new VectorWritable(instance)));
      }
      return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
    }
  }

  @Override
  public double classifyScalar(Vector instance) {
    if (models.size() == 2) {
      double pdf0 = models.get(0).pdf(new VectorWritable(instance));
      double pdf1 = models.get(1).pdf(new VectorWritable(instance));
      return pdf0 / (pdf0 + pdf1);
    }
    throw new IllegalStateException();
  }

  @Override
  public int numCategories() {
    return models.size();
  }
}
