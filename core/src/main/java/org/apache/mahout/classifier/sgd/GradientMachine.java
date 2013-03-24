/*
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

import com.google.common.collect.Sets;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.Random;

/**
 * Online gradient machine learner that tries to minimize the label ranking hinge loss.
 * Implements a gradient machine with one sigmpod hidden layer.
 * It tries to minimize the ranking loss of some given set of labels,
 * so this can be used for multi-class, multi-label
 * or auto-encoding of sparse data (e.g. text).
 */
public class GradientMachine extends AbstractVectorClassifier implements OnlineLearner, Writable {

  public static final int WRITABLE_VERSION = 1;

  // the learning rate of the algorithm
  private double learningRate = 0.1;

  // the regularization term, a positive number that controls the size of the weight vector
  private double regularization = 0.1;

  // the sparsity term, a positive number that controls the sparsity of the hidden layer. (0 - 1)
  private double sparsity = 0.1;

  // the sparsity learning rate.
  private double sparsityLearningRate = 0.1;

  // the number of features
  private int numFeatures = 10;
  // the number of hidden nodes
  private int numHidden = 100;
  // the number of output nodes
  private int numOutput = 2;

  // coefficients for the input to hidden layer.
  // There are numHidden Vectors of dimension numFeatures.
  private Vector[] hiddenWeights;

  // coefficients for the hidden to output layer.
  // There are numOuput Vectors of dimension numHidden.
  private Vector[] outputWeights;

  // hidden unit bias
  private Vector hiddenBias;

  // output unit bias
  private Vector outputBias;

  private final Random rnd;

  public GradientMachine(int numFeatures, int numHidden, int numOutput) {
    this.numFeatures = numFeatures;
    this.numHidden = numHidden;
    this.numOutput = numOutput;
    hiddenWeights = new DenseVector[numHidden];
    for (int i = 0; i < numHidden; i++) {
      hiddenWeights[i] = new DenseVector(numFeatures);
      hiddenWeights[i].assign(0);
    }
    hiddenBias = new DenseVector(numHidden);
    hiddenBias.assign(0);
    outputWeights = new DenseVector[numOutput];
    for (int i = 0; i < numOutput; i++) {
      outputWeights[i] = new DenseVector(numHidden);
      outputWeights[i].assign(0);
    }
    outputBias = new DenseVector(numOutput);
    outputBias.assign(0);
    rnd = RandomUtils.getRandom();
  }

  /**
   * Initialize weights.
   *
   * @param gen random number generator.
   */
  public void initWeights(Random gen) {
    double hiddenFanIn = 1.0 / Math.sqrt(numFeatures);
    for (int i = 0; i < numHidden; i++) {
      for (int j = 0; j < numFeatures; j++) {
        double val = (2.0 * gen.nextDouble() - 1.0) * hiddenFanIn;
        hiddenWeights[i].setQuick(j, val);
      }
    }
    double outputFanIn = 1.0 / Math.sqrt(numHidden);
    for (int i = 0; i < numOutput; i++) {
      for (int j = 0; j < numHidden; j++) {
        double val = (2.0 * gen.nextDouble() - 1.0) * outputFanIn;
        outputWeights[i].setQuick(j, val);
      }
    }
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public GradientMachine learningRate(double learningRate) {
    this.learningRate = learningRate;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param regularization A positive value that controls the weight vector size.
   * @return This, so other configurations can be chained.
   */
  public GradientMachine regularization(double regularization) {
    this.regularization = regularization;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param sparsity A value between zero and one that controls the fraction of hidden units
   *                 that are activated on average.
   * @return This, so other configurations can be chained.
   */
  public GradientMachine sparsity(double sparsity) {
    this.sparsity = sparsity;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param sparsityLearningRate New value of initial learning rate for sparsity.
   * @return This, so other configurations can be chained.
   */
  public GradientMachine sparsityLearningRate(double sparsityLearningRate) {
    this.sparsityLearningRate = sparsityLearningRate;
    return this;
  }

  public void copyFrom(GradientMachine other) {
    numFeatures = other.numFeatures;
    numHidden = other.numHidden;
    numOutput = other.numOutput;
    learningRate = other.learningRate;
    regularization = other.regularization;
    sparsity = other.sparsity;
    sparsityLearningRate = other.sparsityLearningRate;
    hiddenWeights = new DenseVector[numHidden];
    for (int i = 0; i < numHidden; i++) {
      hiddenWeights[i] = other.hiddenWeights[i].clone();
    }
    hiddenBias = other.hiddenBias.clone();
    outputWeights = new DenseVector[numOutput];
    for (int i = 0; i < numOutput; i++) {
      outputWeights[i] = other.outputWeights[i].clone();
    }
    outputBias = other.outputBias.clone();
  }

  @Override
  public int numCategories() {
    return numOutput;
  }

  public int numFeatures() {
    return numFeatures;
  }

  public int numHidden() {
    return numHidden;
  }

  /**
   * Feeds forward from input to hidden unit..
   *
   * @return Hidden unit activations.
   */
  public DenseVector inputToHidden(Vector input) {
    DenseVector activations = new DenseVector(numHidden);
    for (int i = 0; i < numHidden; i++) {
      activations.setQuick(i, hiddenWeights[i].dot(input));
    }
    activations.assign(hiddenBias, Functions.PLUS);
    activations.assign(Functions.min(40.0)).assign(Functions.max(-40));
    activations.assign(Functions.SIGMOID);
    return activations;
  }

  /**
   * Feeds forward from hidden to output
   *
   * @return Output unit activations.
   */
  public DenseVector hiddenToOutput(Vector hiddenActivation) {
    DenseVector activations = new DenseVector(numOutput);
    for (int i = 0; i < numOutput; i++) {
      activations.setQuick(i, outputWeights[i].dot(hiddenActivation));
    }
    activations.assign(outputBias, Functions.PLUS);
    return activations;
  }

  /**
   * Updates using ranking loss.
   *
   * @param hiddenActivation the hidden unit's activation
   * @param goodLabels       the labels you want ranked above others.
   * @param numTrials        how many times you want to search for the highest scoring bad label.
   * @param gen              Random number generator.
   */
  public void updateRanking(Vector hiddenActivation,
                            Collection<Integer> goodLabels,
                            int numTrials,
                            Random gen) {
    // All the labels are good, do nothing.
    if (goodLabels.size() >= numOutput) {
      return;
    }
    for (Integer good : goodLabels) {
      double goodScore = outputWeights[good].dot(hiddenActivation);
      int highestBad = -1;
      double highestBadScore = Double.NEGATIVE_INFINITY;
      for (int i = 0; i < numTrials; i++) {
        int bad = gen.nextInt(numOutput);
        while (goodLabels.contains(bad)) {
          bad = gen.nextInt(numOutput);
        }
        double badScore = outputWeights[bad].dot(hiddenActivation);
        if (badScore > highestBadScore) {
          highestBadScore = badScore;
          highestBad = bad;
        }
      }
      int bad = highestBad;
      double loss = 1.0 - goodScore + highestBadScore;
      if (loss < 0.0) {
        continue;
      }
      // Note from the loss above the gradient dloss/dy , y being the label is -1 for good
      // and +1 for bad.
      // dy / dw is just w since  y = x' * w + b.
      // Hence by the chain rule, dloss / dw = dloss / dy * dy / dw = -w.
      // For the regularization part, 0.5 * lambda * w' w, the gradient is lambda * w.
      // dy / db = 1.
      Vector gradGood = outputWeights[good].clone();
      gradGood.assign(Functions.NEGATE);
      Vector propHidden = gradGood.clone();
      Vector gradBad = outputWeights[bad].clone();
      propHidden.assign(gradBad, Functions.PLUS);
      gradGood.assign(Functions.mult(-learningRate * (1.0 - regularization)));
      outputWeights[good].assign(gradGood, Functions.PLUS);
      gradBad.assign(Functions.mult(-learningRate * (1.0 + regularization)));
      outputWeights[bad].assign(gradBad, Functions.PLUS);
      outputBias.setQuick(good, outputBias.get(good) + learningRate);
      outputBias.setQuick(bad, outputBias.get(bad) - learningRate);
      // Gradient of sigmoid is s * (1 -s).
      Vector gradSig = hiddenActivation.clone();
      gradSig.assign(Functions.SIGMOIDGRADIENT);
      // Multiply by the change caused by the ranking loss.
      for (int i = 0; i < numHidden; i++) {
        gradSig.setQuick(i, gradSig.get(i) * propHidden.get(i));
      }
      for (int i = 0; i < numHidden; i++) {
        for (int j = 0; j < numFeatures; j++) {
          double v = hiddenWeights[i].get(j);
          v -= learningRate * (gradSig.get(i) + regularization * v);
          hiddenWeights[i].setQuick(j, v);
        }
      }
    }
  }

  @Override
  public Vector classify(Vector instance) {
    Vector result = classifyNoLink(instance);
    // Find the max value's index.
    int max = result.maxValueIndex();
    result.assign(0);
    result.setQuick(max, 1.0);
    return result.viewPart(1, result.size() - 1);
  }

  @Override
  public Vector classifyNoLink(Vector instance) {
    DenseVector hidden = inputToHidden(instance);
    return hiddenToOutput(hidden);
  }

  @Override
  public double classifyScalar(Vector instance) {
    Vector output = classifyNoLink(instance);
    if (output.get(0) > output.get(1)) {
      return 0;
    }
    return 1;
  }

  public GradientMachine copy() {
    close();
    GradientMachine r = new GradientMachine(numFeatures(), numHidden(), numCategories());
    r.copyFrom(this);
    return r;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(WRITABLE_VERSION);
    out.writeDouble(learningRate);
    out.writeDouble(regularization);
    out.writeDouble(sparsity);
    out.writeDouble(sparsityLearningRate);
    out.writeInt(numFeatures);
    out.writeInt(numHidden);
    out.writeInt(numOutput);
    VectorWritable.writeVector(out, hiddenBias);
    for (int i = 0; i < numHidden; i++) {
      VectorWritable.writeVector(out, hiddenWeights[i]);
    }
    VectorWritable.writeVector(out, outputBias);
    for (int i = 0; i < numOutput; i++) {
      VectorWritable.writeVector(out, outputWeights[i]);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int version = in.readInt();
    if (version == WRITABLE_VERSION) {
      learningRate = in.readDouble();
      regularization = in.readDouble();
      sparsity = in.readDouble();
      sparsityLearningRate = in.readDouble();
      numFeatures = in.readInt();
      numHidden = in.readInt();
      numOutput = in.readInt();
      hiddenWeights = new DenseVector[numHidden];
      hiddenBias = VectorWritable.readVector(in);
      for (int i = 0; i < numHidden; i++) {
        hiddenWeights[i] = VectorWritable.readVector(in);
      }
      outputWeights = new DenseVector[numOutput];
      outputBias = VectorWritable.readVector(in);
      for (int i = 0; i < numOutput; i++) {
        outputWeights[i] = VectorWritable.readVector(in);
      }
    } else {
      throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
    }
  }

  @Override
  public void close() {
    // This is an online classifier, nothing to do.
  }

  @Override
  public void train(long trackingKey, String groupKey, int actual, Vector instance) {
    Vector hiddenActivation = inputToHidden(instance);
    hiddenToOutput(hiddenActivation);
    Collection<Integer> goodLabels = Sets.newHashSet();
    goodLabels.add(actual);
    updateRanking(hiddenActivation, goodLabels, 2, rnd);
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(int actual, Vector instance) {
    train(0, null, actual, instance);
  }

}
