package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Lists;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.OnlineAuc;

import java.util.List;

/**
 * Does cross-fold validation of log-likelihood and AUC on several online logistic regression
 * models. Each record is passed to all but one of the models for training and to the remaining
 * model for evaluation.  In order to maintain proper segregation between the different folds across
 * training data iterations, data should either be passed to this learner in the same order each
 * time the training data is traversed or a tracking key such as the file offset of the training
 * record should be passed with each training example.
 */
class CrossFoldLearner extends AbstractVectorClassifier implements OnlineLearner, Comparable<CrossFoldLearner> {
  private static volatile int nextId = 0;

  private final int id = nextId++;
  private int record = 0;
  private OnlineAuc auc = new OnlineAuc();
  private double logLikelihood = 0;
  private List<OnlineLogisticRegression> models = Lists.newArrayList();

  // lambda, learningRate, perTermOffset, perTermExponent
  private double[] parameters = new double[4];

  CrossFoldLearner(int folds, int numCategories, int numFeatures, PriorFunction prior) {
    for (int i = 0; i < folds; i++) {
      OnlineLogisticRegression model = new OnlineLogisticRegression(numCategories, numFeatures, prior);
      model.alpha(1).stepOffset(0).decayExponent(0);
      models.add(model);
    }
  }

  // -------- builder-like configuration methods

  public CrossFoldLearner lambda(double v) {
    for (OnlineLogisticRegression model : models) {
      model.lambda(v);
    }
    return this;
  }

  public CrossFoldLearner learningRate(double x) {
    for (OnlineLogisticRegression model : models) {
      model.learningRate(x);
    }
    return this;
  }

  public CrossFoldLearner stepOffset(int x) {
    for (OnlineLogisticRegression model : models) {
      model.stepOffset(x);
    }
    return this;
  }

  public CrossFoldLearner decayExponent(double x) {
    for (OnlineLogisticRegression model : models) {
      model.decayExponent(x);
    }
    return this;
  }

  // -------- training methods
  @Override
  public void train(int actual, Vector instance) {
    train(record, actual, instance);
  }

  @Override
  public void train(int trackingKey, int actual, Vector instance) {
    record++;
    int k = 0;
    for (OnlineLogisticRegression model : models) {
      if (k == trackingKey % models.size()) {
        Vector v = model.classifyFull(instance);
        double score = v.get(actual);
        logLikelihood += (Math.log(score) - logLikelihood) / record;
        auc.addSample(actual, v.get(1));
      } else {
        model.train(actual, instance);
      }
      k = (k + 1) % models.size();
    }
  }

  @Override
  public void close() {
    for (OnlineLogisticRegression m : models) {
      m.close();
    }
  }

  public void resetLineCounter() {
    record = 0;
  }

  // -------- classification methods

  @Override
  public Vector classify(Vector instance) {
    Vector r = new DenseVector(numCategories() - 1);
    double scale = 1.0 / models.size();
    for (OnlineLogisticRegression model : models) {
      r.assign(model.classify(instance), Functions.plusMult(scale));
    }
    return r;
  }

  @Override
  public double classifyScalar(Vector instance) {
    double r = 0;
    int n = 0;
    for (OnlineLogisticRegression model : models) {
      n++;
      r += model.classifyScalar(instance);
    }
    return r / n;
  }

  // -------- status reporting methods
  
  @Override
  public int numCategories() {
    return models.get(0).numCategories();
  }

  public double auc() {
    return auc.auc();
  }

  public double logLikelihood() {
    return logLikelihood;
  }
  // -------- general object and ordering stuff

  /**
   * Orders primarily by AUC descending with a fallback to object creation order so that
   * <pre>
   * a.compareTo(b) == 0  <==>  a.equals(b)
   * </pre>
   * Without the id comparison, it is dangerous to insert CrossFoldLearner's into a
   * TreeSet since if they ever have the same score, they can't be distinguished.
   */
  @Override
  public int compareTo(CrossFoldLearner other) {
    int r = Double.compare(this.auc(), other.auc());
    if (r != 0) {
      return r;
    } else {
      return id - other.id;
    }
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof CrossFoldLearner && id == ((CrossFoldLearner) other).id;
  }
}
