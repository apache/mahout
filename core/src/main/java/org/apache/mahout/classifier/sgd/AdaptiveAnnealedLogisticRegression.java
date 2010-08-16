package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Lists;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.random.NegativeBinomial;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;

import java.util.Collections;
import java.util.List;

/**
 * This is a meta-learner that maintains a pool of ordinary OnlineLogisticRegression learners. Each
 * member of the pool has different learning rates.  Whichever of the learners in the pool falls
 * behind in terms of average log-likelihood will be tossed out and replaced with variants of the
 * survivors.  This will let us automatically derive an annealing schedule that optimizes learning
 * speed.  Since on-line learners tend to be IO bound anyway, it doesn't cost as much as it might
 * seem that it would to maintain multiple learners in memory.  Doing this adaptation on-line as we
 * learn also decreases the number of learning rate parameters required and replaces the normal
 * hyper-parameter search.
 */
public class AdaptiveAnnealedLogisticRegression   implements OnlineLearner {
  private int record = 0;
  private List<CrossFoldLearner> pool = Lists.newArrayList();
  private int evaluationInterval = 1000;

  public AdaptiveAnnealedLogisticRegression(int poolSize, int numCategories, int numFeatures, PriorFunction prior) {
    for (int i = 0; i < poolSize; i++) {
      CrossFoldLearner model = new CrossFoldLearner(5, numCategories, numFeatures, prior);
      pool.add(model);
    }
    NegativeBinomial nb = new NegativeBinomial(10, 0.1, new MersenneTwister());
  }

  @Override
  public void train(int actual, Vector instance) {
    for (CrossFoldLearner learner : pool) {
      learner.train(actual, instance);
    }
    record++;
    if (record % evaluationInterval == 0) {
      Collections.sort(pool);
      // pick a parent from the top half of the pool weighted toward the top few

    }
  }

  @Override
  public void train(int trackingKey, int actual, Vector instance) {
    train(actual, instance);
  }

  @Override
  public void close() {
    //To change body of implemented methods use File | Settings | File Templates.
  }

}
