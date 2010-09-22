/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Lists;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.ep.Payload;
import org.apache.mahout.ep.EvolutionaryProcess;
import org.apache.mahout.ep.Mapping;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

/**
 * This is a meta-learner that maintains a pool of ordinary OnlineLogisticRegression learners. Each
 * member of the pool has different learning rates.  Whichever of the learners in the pool falls
 * behind in terms of average log-likelihood will be tossed out and replaced with variants of the
 * survivors.  This will let us automatically derive an annealing schedule that optimizes learning
 * speed.  Since on-line learners tend to be IO bound anyway, it doesn't cost as much as it might
 * seem that it would to maintain multiple learners in memory.  Doing this adaptation on-line as we
 * learn also decreases the number of learning rate parameters required and replaces the normal
 * hyper-parameter search.
 * <p/>
 * One wrinkle is that the pool of learners that we maintain is actually a pool of CrossFoldLearners
 * which themselves contain several OnlineLogisticRegression objects.  These pools allow estimation
 * of performance on the fly even if we make many passes through the data.  This does, however,
 * increase the cost of training since if we are using 5-fold cross-validation, each vector is used
 * 4 times for training and once for classification.  If this becomes a problem, then we should
 * probably use a 2-way unbalanced train/test split rather than full cross validation.  With the
 * current default settings, we have 100 learners running.  This is better than the alternative
 * of running hundreds of training passes to find good hyper-parameters because we only have to
 * parse and feature-ize our inputs once.  If you already have good hyper-parameters, then you
 * might prefer to just run one CrossFoldLearner with those settings.
 * <p/>
 * The fitness used here is AUC.  Another alternative would be to try log-likelihood, but it is
 * much easier to get bogus values of log-likelihood than with AUC and the results seem to
 * accord pretty well.  It would be nice to allow the fitness function to be pluggable. This
 * use of AUC means that AdaptiveLogisticRegression is mostly suited for binary target variables.
 * This will be fixed before long by extending OnlineAuc to handle non-binary cases or by using
 * a different fitness value in non-binary cases.
 */
public class AdaptiveLogisticRegression implements OnlineLearner {
  private static final int SURVIVORS = 2;

  private int record = 0;
  private int evaluationInterval = 1000;

  private List<TrainingExample> buffer = Lists.newArrayList();
  private EvolutionaryProcess<Wrapper> ep;
  private State<Wrapper> best;
  private int threadCount = 20;
  private int poolSize = 20;
  private State<Wrapper> seed;
  private int numFeatures;
  private double averagingWindow;

  private boolean freezeSurvivors = true;

  // for GSON
  private AdaptiveLogisticRegression() {
  }

  public AdaptiveLogisticRegression(int numCategories, int numFeatures, PriorFunction prior) {
    this.numFeatures = numFeatures;
    seed = new State<Wrapper>(new double[2], 10);
    Wrapper w = new Wrapper(numCategories, numFeatures, prior);
    w.setMappings(seed);
    seed.setPayload(w);
    setPoolSize(poolSize);
  }

  @Override
  public void train(int actual, Vector instance) {
    train(record, actual, instance);
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    record++;

    buffer.add(new TrainingExample(trackingKey, actual, instance));
    if (buffer.size() > evaluationInterval) {
      trainWithBufferedExamples();
    }
  }

  private void trainWithBufferedExamples() {
    try {
      this.best = ep.parallelDo(new EvolutionaryProcess.Function<Wrapper>() {
        @Override
        public double apply(Wrapper x, double[] params) {
          for (TrainingExample example : buffer) {
            x.train(example);
          }
          if (x.getLearner().validModel()) {
            if (x.getLearner().numCategories() == 2) {
              return x.wrapped.auc();
            } else {
              return x.wrapped.logLikelihood();
            }
          } else {
            return Double.NaN;
          }
        }
      });
    } catch (InterruptedException e) {
      // ignore ... shouldn't happen
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }

    // evolve based on new fitness
    ep.mutatePopulation(SURVIVORS);

    if (freezeSurvivors) {
      // now grossly hack the top survivors so they stick around.  Set their
      // mutation rates small and also hack their learning rate to be small
      // as well.
      for (State<Wrapper> state : ep.getPopulation().subList(0, SURVIVORS)) {
        state.getPayload().freeze(state);
      }
    }
    buffer.clear();
  }

  @Override
  public void close() {
    trainWithBufferedExamples();
    try {
      ep.parallelDo(new EvolutionaryProcess.Function<Wrapper>() {
        @Override
        public double apply(Wrapper payload, double[] params) {
          payload.getLearner().close();
          return payload.getLearner().logLikelihood();
        }
      });
      ep.close();
    } catch (InterruptedException e) {
      // ignore
    } catch (ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * How often should the evolutionary optimization of learning parameters occur?
   * @param interval  Number of training examples to use in each epoch of optimization.
   */
  public void setInterval(int interval) {
    this.evaluationInterval = interval;
  }

  public void setPoolSize(int poolSize) {
    this.poolSize = poolSize;
    setupOptimizer(poolSize);
  }

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
    setupOptimizer(poolSize);
  }

  private void setupOptimizer(int poolSize) {
    ep = new EvolutionaryProcess<Wrapper>(threadCount, poolSize, seed);
  }

  /**
   * Returns the size of the internal feature vector.  Note that this is not the
   * same as the number of distinct features, especially if feature hashing is
   * being used.
   * @return The internal feature vector size.
   */
  public int numFeatures() {
    return numFeatures;
  }

  /**
   * What is the AUC for the current best member of the population.  If no member is best,
   * usually because we haven't done any training yet, then the result is set to NaN.
   * @return  The AUC of the best member of the population or NaN if we can't figure that out.
   */
  public double auc() {
    if (best == null) {
      return Double.NaN;
    } else {
      return best.getPayload().getLearner().auc();
    }
  }

  public State<Wrapper> getBest() {
    return best;
  }

  public void setBest(State<Wrapper> best) {
    this.best = best;
  }

  public int getRecord() {
    return record;
  }

  public void setRecord(int record) {
    this.record = record;
  }

  public int getEvaluationInterval() {
    return evaluationInterval;
  }

  public int getNumCategories() {
    return seed.getPayload().getLearner().numCategories();
  }

  public PriorFunction getPrior() {
    return seed.getPayload().getLearner().getPrior();
  }

  public void setEvaluationInterval(int evaluationInterval) {
    this.evaluationInterval = evaluationInterval;
  }

  public void setBuffer(List<TrainingExample> buffer) {
    this.buffer = buffer;
  }

  public List<TrainingExample> getBuffer() {
    return buffer;
  }

  public EvolutionaryProcess<Wrapper> getEp() {
    return ep;
  }

  public void setEp(EvolutionaryProcess<Wrapper> ep) {
    this.ep = ep;
  }

  public State<Wrapper> getSeed() {
    return seed;
  }

  public void setSeed(State<Wrapper> seed) {
    this.seed = seed;
  }

  public int getNumFeatures() {
    return numFeatures;
  }

  public void setNumFeatures(int numFeatures) {
    this.numFeatures = numFeatures;
  }

  public void setAveragingWindow(int averagingWindow) {
    seed.getPayload().getLearner().setWindowSize(averagingWindow);
  }

  public void setFreezeSurvivors(boolean freezeSurvivors) {
    this.freezeSurvivors = freezeSurvivors;
  }

  /**
   * Provides a shim between the EP optimization stuff and the CrossFoldLearner.  The most important
   * interface has to do with the parameters of the optimization.  These are taken from the double[]
   * params in the following order <ul> <li> regularization constant lambda <li> learningRate </ul>.
   * All other parameters are set in such a way so as to defeat annealing to the extent possible.
   * This lets the evolutionary algorithm handle the annealing.
   * <p>
   * Note that per coefficient annealing is still done and no optimization of the per coefficient
   * offset is done.
   */
  public static class Wrapper implements Payload<Wrapper> {
    private CrossFoldLearner wrapped;

    private Wrapper() {
      // just here to help copy
    }

    public Wrapper(int numCategories, int numFeatures, PriorFunction prior) {
      wrapped = new CrossFoldLearner(5, numCategories, numFeatures, prior);
    }

    @Override
    public Wrapper copy() {
      Wrapper r = new Wrapper();
      r.wrapped = wrapped.copy();
      return r;
    }

    @Override
    public void update(double[] params) {
      int i = 0;
      wrapped.lambda(params[i++]);
      wrapped.learningRate(params[i]);

      wrapped.stepOffset(1);
      wrapped.alpha(1);
      wrapped.decayExponent(0);
    }

    public void freeze(State<Wrapper> s) {
      // radically decrease learning rate
      s.getParams()[1] -= 10;

      // and cause evolution to hold (almost)
      s.setOmni(s.getOmni() / 20);
      double[] step = s.getStep();
      for (int i = 0; i < step.length; i++) {
        step[i] /= 20;
      }
    }

    public void setMappings(State<Wrapper> x) {
      int i = 0;
      // set the range for regularization (lambda)
      x.setMap(i++, Mapping.logLimit(1.0e-8, 0.1));
      // set the range for learning rate (mu)
      x.setMap(i, Mapping.softLimit(0.001, 10));
    }

    public void train(TrainingExample example) {
      wrapped.train(example.getKey(), example.getActual(), example.getInstance());
    }

    public CrossFoldLearner getLearner() {
      return wrapped;
    }

    @Override
    public String toString() {
      return String.format(Locale.ENGLISH, "auc=%.2f", wrapped.auc());
    }
  }

  public static class TrainingExample {
    private long key;
    private int actual;
    private Vector instance;

    // for GSON
    private TrainingExample() {
    }

    public TrainingExample(long key, int actual, Vector instance) {
      this.key = key;
      this.actual = actual;
      this.instance = instance;
    }

    public long getKey() {
      return key;
    }

    public int getActual() {
      return actual;
    }

    public Vector getInstance() {
      return instance;
    }
  }
}

