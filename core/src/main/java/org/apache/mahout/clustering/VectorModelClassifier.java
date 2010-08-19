package org.apache.mahout.clustering;

import java.util.List;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

public class VectorModelClassifier extends AbstractVectorClassifier {

  List<Model<VectorWritable>> models;

  public VectorModelClassifier(List<Model<VectorWritable>> models) {
    super();
    this.models = models;
  }

  @Override
  public Vector classify(Vector instance) {
    Vector pdfs = new DenseVector(models.size());
    int i = 0;
    for (Model<VectorWritable> model : models) {
      pdfs.set(i++, model.pdf(new VectorWritable(instance)));
    }
    return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
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
