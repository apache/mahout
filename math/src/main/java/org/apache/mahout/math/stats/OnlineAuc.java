package org.apache.mahout.math.stats;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.Random;

/**
 * Computes a running estimate of AUC (see http://en.wikipedia.org/wiki/Receiver_operating_characteristic).
 * <p/>
 * Since AUC is normally a global property of labeled scores, it is almost always computed in a
 * batch fashion.  The probabilistic definition (the probability that a random element of one set
 * has a higher score than a random element of another set) gives us a way to estimate this
 * on-line.
 */
public class OnlineAuc {
  private Random random = new Random();

  enum ReplacementPolicy {
    FIFO, FAIR, RANDOM
  }

  public static final int HISTORY = 10;

  private ReplacementPolicy policy = ReplacementPolicy.FAIR;

  private Matrix scores;
  private Vector averages;

  private Vector samples;

  public OnlineAuc() {
    int numCategories = 2;
    scores = new DenseMatrix(numCategories, HISTORY);
    scores.assign(Double.NaN);
    averages = new DenseVector(numCategories);
    averages.assign(0.5);
    samples = new DenseVector(numCategories);
  }

  public double addSample(int category, final double score) {
    int n = (int) samples.get(category);
    if (n < HISTORY) {
      scores.set(category, n, score);
    } else {
      switch (policy) {
        case FIFO:
          scores.set(category, n % HISTORY, score);
          break;
        case FAIR:
          int j = random.nextInt(n + 1);
          if (j < HISTORY) {
            scores.set(category, j, score);
          }
          break;
        case RANDOM:
          j = random.nextInt(HISTORY);
          scores.set(category, j, score);
          break;
      }
    }

    samples.set(category, n + 1);

    if (samples.minValue() >= 1) {
      // compare to previous scores for other category
      Vector row = scores.viewRow(1 - category);
      double m = 0;
      int count = 0;
      for (Vector.Element element : row) {
        double v = element.get();
        if (!Double.isNaN(v)) {
          count++;
          double z = 0.5;
          if (score > v) {
            z = 1;
          } else if (score < v) {
            z = 0;
          }
          m += (z - m) / count;
        } else {
          break;
        }
      }
      averages.set(category, averages.get(category) + (m - averages.get(category)) / samples.get(category));
    }
    return auc();
  }

  public double auc() {
    // return an unweighted average of all averages.
    return 0.5 - averages.get(0) / 2 + averages.get(1) / 2;
  }

  public void setPolicy(ReplacementPolicy policy) {
    this.policy = policy;
  }

  public void setRandom(Random random) {
    this.random = random;
  }
}
