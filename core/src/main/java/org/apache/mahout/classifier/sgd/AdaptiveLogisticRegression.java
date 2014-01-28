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
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.ep.EvolutionaryProcess;
import org.apache.mahout.ep.Mapping;
import org.apache.mahout.ep.Payload;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.OnlineAuc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

/**
 * This is a meta-learner that maintains a pool of ordinary
 * {@link org.apache.mahout.classifier.sgd.OnlineLogisticRegression} learners. Each
 * member of the pool has different learning rates.  Whichever of the learners in the pool falls
 * behind in terms of average log-likelihood will be tossed out and replaced with variants of the
 * survivors.  This will let us automatically derive an annealing schedule that optimizes learning
 * speed.  Since on-line learners tend to be IO bound anyway, it doesn't cost as much as it might
 * seem that it would to maintain multiple learners in memory.  Doing this adaptation on-line as we
 * learn also decreases the number of learning rate parameters required and replaces the normal
 * hyper-parameter search.
 * <p/>
 * One wrinkle is that the pool of learners that we maintain is actually a pool of
 * {@link org.apache.mahout.classifier.sgd.CrossFoldLearner} which themselves contain several OnlineLogisticRegression
 * objects.  These pools allow estimation
 * of performance on the fly even if we make many passes through the data.  This does, however,
 * increase the cost of training since if we are using 5-fold cross-validation, each vector is used
 * 4 times for training and once for classification.  If this becomes a problem, then we should
 * probably use a 2-way unbalanced train/test split rather than full cross validation.  With the
 * current default settings, we have 100 learners running.  This is better than the alternative of
 * running hundreds of training passes to find good hyper-parameters because we only have to parse
 * and feature-ize our inputs once.  If you already have good hyper-parameters, then you might
 * prefer to just run one CrossFoldLearner with those settings.
 * <p/>
 * The fitness used here is AUC.  Another alternative would be to try log-likelihood, but it is much
 * easier to get bogus values of log-likelihood than with AUC and the results seem to accord pretty
 * well.  It would be nice to allow the fitness function to be pluggable. This use of AUC means that
 * AdaptiveLogisticRegression is mostly suited for binary target variables. This will be fixed
 * before long by extending OnlineAuc to handle non-binary cases or by using a different fitness
 * value in non-binary cases.
 */
public class AdaptiveLogisticRegression implements OnlineLearner, Writable {
  public static final int DEFAULT_THREAD_COUNT = 20;
  public static final int DEFAULT_POOL_SIZE = 20;
  private static final int SURVIVORS = 2;

  private int record;
  private int cutoff = 1000;
  private int minInterval = 1000;
  private int maxInterval = 1000;
  private int currentStep = 1000;
  private int bufferSize = 1000;

  private List<TrainingExample> buffer = Lists.newArrayList();
  private EvolutionaryProcess<Wrapper, CrossFoldLearner> ep;
  private State<Wrapper, CrossFoldLearner> best;
  private int threadCount = DEFAULT_THREAD_COUNT;
  private int poolSize = DEFAULT_POOL_SIZE;
  private State<Wrapper, CrossFoldLearner> seed;
  private int numFeatures;

  private boolean freezeSurvivors = true;

  private static final Logger log = LoggerFactory.getLogger(AdaptiveLogisticRegression.class);

  public AdaptiveLogisticRegression() {}

  /**
   * Uses {@link #DEFAULT_THREAD_COUNT} and {@link #DEFAULT_POOL_SIZE}
   * @param numCategories The number of categories (labels) to train on
   * @param numFeatures The number of features used in creating the vectors (i.e. the cardinality of the vector)
   * @param prior The {@link org.apache.mahout.classifier.sgd.PriorFunction} to use
   *
   * @see #AdaptiveLogisticRegression(int, int, org.apache.mahout.classifier.sgd.PriorFunction, int, int)
   */
  public AdaptiveLogisticRegression(int numCategories, int numFeatures, PriorFunction prior) {
    this(numCategories, numFeatures, prior, DEFAULT_THREAD_COUNT, DEFAULT_POOL_SIZE);
  }

  /**
   *
   * @param numCategories The number of categories (labels) to train on
   * @param numFeatures The number of features used in creating the vectors (i.e. the cardinality of the vector)
   * @param prior The {@link org.apache.mahout.classifier.sgd.PriorFunction} to use
   * @param threadCount The number of threads to use for training
   * @param poolSize The number of {@link org.apache.mahout.classifier.sgd.CrossFoldLearner} to use.
   */
  public AdaptiveLogisticRegression(int numCategories, int numFeatures, PriorFunction prior, int threadCount,
      int poolSize) {
    this.numFeatures = numFeatures;
    this.threadCount = threadCount;
    this.poolSize = poolSize;
    seed = new State<Wrapper, CrossFoldLearner>(new double[2], 10);
    Wrapper w = new Wrapper(numCategories, numFeatures, prior);
    seed.setPayload(w);

    Wrapper.setMappings(seed);
    seed.setPayload(w);
    setPoolSize(this.poolSize);
  }

  @Override
  public void train(int actual, Vector instance) {
    train(record, null, actual, instance);
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(long trackingKey, String groupKey, int actual, Vector instance) {
    record++;

    buffer.add(new TrainingExample(trackingKey, groupKey, actual, instance));
    //don't train until we have enough examples
    if (buffer.size() > bufferSize) {
      trainWithBufferedExamples();
    }
  }

  private void trainWithBufferedExamples() {
    try {
      this.best = ep.parallelDo(new EvolutionaryProcess.Function<Payload<CrossFoldLearner>>() {
        @Override
        public double apply(Payload<CrossFoldLearner> z, double[] params) {
          Wrapper x = (Wrapper) z;
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
      log.warn("Ignoring exception", e);
    } catch (ExecutionException e) {
      throw new IllegalStateException(e.getCause());
    }
    buffer.clear();

    if (record > cutoff) {
      cutoff = nextStep(record);

      // evolve based on new fitness
      ep.mutatePopulation(SURVIVORS);

      if (freezeSurvivors) {
        // now grossly hack the top survivors so they stick around.  Set their
        // mutation rates small and also hack their learning rate to be small
        // as well.
        for (State<Wrapper, CrossFoldLearner> state : ep.getPopulation().subList(0, SURVIVORS)) {
          Wrapper.freeze(state);
        }
      }
    }

  }

  public int nextStep(int recordNumber) {
    int stepSize = stepSize(recordNumber, 2.6);
    if (stepSize < minInterval) {
      stepSize = minInterval;
    }

    if (stepSize > maxInterval) {
      stepSize = maxInterval;
    }

    int newCutoff = stepSize * (recordNumber / stepSize + 1);
    if (newCutoff < cutoff + currentStep) {
      newCutoff = cutoff + currentStep;
    } else {
      this.currentStep = stepSize;
    }
    return newCutoff;
  }

  public static int stepSize(int recordNumber, double multiplier) {
    int[] bumps = {1, 2, 5};
    double log = Math.floor(multiplier * Math.log10(recordNumber));
    int bump = bumps[(int) log % bumps.length];
    int scale = (int) Math.pow(10, Math.floor(log / bumps.length));

    return bump * scale;
  }

  @Override
  public void close() {
    trainWithBufferedExamples();
    try {
      ep.parallelDo(new EvolutionaryProcess.Function<Payload<CrossFoldLearner>>() {
        @Override
        public double apply(Payload<CrossFoldLearner> payload, double[] params) {
          CrossFoldLearner learner = ((Wrapper) payload).getLearner();
          learner.close();
          return learner.logLikelihood();
        }
      });
    } catch (InterruptedException e) {
      log.warn("Ignoring exception", e);
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    } finally {
      ep.close();
    }
  }

  /**
   * How often should the evolutionary optimization of learning parameters occur?
   *
   * @param interval Number of training examples to use in each epoch of optimization.
   */
  public void setInterval(int interval) {
    setInterval(interval, interval);
  }

  /**
   * Starts optimization using the shorter interval and progresses to the longer using the specified
   * number of steps per decade.  Note that values < 200 are not accepted.  Values even that small
   * are unlikely to be useful.
   *
   * @param minInterval The minimum epoch length for the evolutionary optimization
   * @param maxInterval The maximum epoch length
   */
  public void setInterval(int minInterval, int maxInterval) {
    this.minInterval = Math.max(200, minInterval);
    this.maxInterval = Math.max(200, maxInterval);
    this.cutoff = minInterval * (record / minInterval + 1);
    this.currentStep = minInterval;
    bufferSize = Math.min(minInterval, bufferSize);
  }

  public void setPoolSize(int poolSize) {
    this.poolSize = poolSize;
    setupOptimizer(poolSize);
  }

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
    setupOptimizer(poolSize);
  }

  public void setAucEvaluator(OnlineAuc auc) {
    seed.getPayload().setAucEvaluator(auc);
    setupOptimizer(poolSize);
  }

  private void setupOptimizer(int poolSize) {
    ep = new EvolutionaryProcess<Wrapper, CrossFoldLearner>(threadCount, poolSize, seed);
  }

  /**
   * Returns the size of the internal feature vector.  Note that this is not the same as the number
   * of distinct features, especially if feature hashing is being used.
   *
   * @return The internal feature vector size.
   */
  public int numFeatures() {
    return numFeatures;
  }

  /**
   * What is the AUC for the current best member of the population.  If no member is best, usually
   * because we haven't done any training yet, then the result is set to NaN.
   *
   * @return The AUC of the best member of the population or NaN if we can't figure that out.
   */
  public double auc() {
    if (best == null) {
      return Double.NaN;
    } else {
      Wrapper payload = best.getPayload();
      return payload.getLearner().auc();
    }
  }

  public State<Wrapper, CrossFoldLearner> getBest() {
    return best;
  }

  public void setBest(State<Wrapper, CrossFoldLearner> best) {
    this.best = best;
  }

  public int getRecord() {
    return record;
  }

  public void setRecord(int record) {
    this.record = record;
  }

  public int getMinInterval() {
    return minInterval;
  }

  public int getMaxInterval() {
    return maxInterval;
  }

  public int getNumCategories() {
    return seed.getPayload().getLearner().numCategories();
  }

  public PriorFunction getPrior() {
    return seed.getPayload().getLearner().getPrior();
  }

  public void setBuffer(List<TrainingExample> buffer) {
    this.buffer = buffer;
  }

  public List<TrainingExample> getBuffer() {
    return buffer;
  }

  public EvolutionaryProcess<Wrapper, CrossFoldLearner> getEp() {
    return ep;
  }

  public void setEp(EvolutionaryProcess<Wrapper, CrossFoldLearner> ep) {
    this.ep = ep;
  }

  public State<Wrapper, CrossFoldLearner> getSeed() {
    return seed;
  }

  public void setSeed(State<Wrapper, CrossFoldLearner> seed) {
    this.seed = seed;
  }

  public int getNumFeatures() {
    return numFeatures;
  }

  public void setAveragingWindow(int averagingWindow) {
    seed.getPayload().getLearner().setWindowSize(averagingWindow);
    setupOptimizer(poolSize);
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
   * <p/>
   * Note that per coefficient annealing is still done and no optimization of the per coefficient
   * offset is done.
   */
  public static class Wrapper implements Payload<CrossFoldLearner> {
    private CrossFoldLearner wrapped;

    public Wrapper() {
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

    public static void freeze(State<Wrapper, CrossFoldLearner> s) {
      // radically decrease learning rate
      double[] params = s.getParams();
      params[1] -= 10;

      // and cause evolution to hold (almost)
      s.setOmni(s.getOmni() / 20);
      double[] step = s.getStep();
      for (int i = 0; i < step.length; i++) {
        step[i] /= 20;
      }
    }

    public static void setMappings(State<Wrapper, CrossFoldLearner> x) {
      int i = 0;
      // set the range for regularization (lambda)
      x.setMap(i++, Mapping.logLimit(1.0e-8, 0.1));
      // set the range for learning rate (mu)
      x.setMap(i, Mapping.logLimit(1.0e-8, 1));
    }

    public void train(TrainingExample example) {
      wrapped.train(example.getKey(), example.getGroupKey(), example.getActual(), example.getInstance());
    }

    public CrossFoldLearner getLearner() {
      return wrapped;
    }

    @Override
    public String toString() {
      return String.format(Locale.ENGLISH, "auc=%.2f", wrapped.auc());
    }

    public void setAucEvaluator(OnlineAuc auc) {
      wrapped.setAucEvaluator(auc);
    }

    @Override
    public void write(DataOutput out) throws IOException {
      wrapped.write(out);
    }

    @Override
    public void readFields(DataInput input) throws IOException {
      wrapped = new CrossFoldLearner();
      wrapped.readFields(input);
    }
  }

  public static class TrainingExample implements Writable {
    private long key;
    private String groupKey;
    private int actual;
    private Vector instance;

    private TrainingExample() {
    }

    public TrainingExample(long key, String groupKey, int actual, Vector instance) {
      this.key = key;
      this.groupKey = groupKey;
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

    public String getGroupKey() {
      return groupKey;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeLong(key);
      if (groupKey != null) {
        out.writeBoolean(true);
        out.writeUTF(groupKey);
      } else {
        out.writeBoolean(false);
      }
      out.writeInt(actual);
      VectorWritable.writeVector(out, instance, true);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      key = in.readLong();
      if (in.readBoolean()) {
        groupKey = in.readUTF();
      }
      actual = in.readInt();
      instance = VectorWritable.readVector(in);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(record);
    out.writeInt(cutoff);
    out.writeInt(minInterval);
    out.writeInt(maxInterval);
    out.writeInt(currentStep);
    out.writeInt(bufferSize);

    out.writeInt(buffer.size());
    for (TrainingExample example : buffer) {
      example.write(out);
    }

    ep.write(out);

    best.write(out);

    out.writeInt(threadCount);
    out.writeInt(poolSize);
    seed.write(out);
    out.writeInt(numFeatures);

    out.writeBoolean(freezeSurvivors);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    record = in.readInt();
    cutoff = in.readInt();
    minInterval = in.readInt();
    maxInterval = in.readInt();
    currentStep = in.readInt();
    bufferSize = in.readInt();

    int n = in.readInt();
    buffer = Lists.newArrayList();
    for (int i = 0; i < n; i++) {
      TrainingExample example = new TrainingExample();
      example.readFields(in);
      buffer.add(example);
    }

    ep = new EvolutionaryProcess<Wrapper, CrossFoldLearner>();
    ep.readFields(in);

    best = new State<Wrapper, CrossFoldLearner>();
    best.readFields(in);

    threadCount = in.readInt();
    poolSize = in.readInt();
    seed = new State<Wrapper, CrossFoldLearner>();
    seed.readFields(in);

    numFeatures = in.readInt();
    freezeSurvivors = in.readBoolean();
  }
}

