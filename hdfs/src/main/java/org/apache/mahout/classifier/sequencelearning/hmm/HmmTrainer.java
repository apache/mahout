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

package org.apache.mahout.classifier.sequencelearning.hmm;

import java.util.Collection;
import java.util.Iterator;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Class containing several algorithms used to train a Hidden Markov Model. The
 * three main algorithms are: supervised learning, unsupervised Viterbi and
 * unsupervised Baum-Welch.
 */
public final class HmmTrainer {

  /**
   * No public constructor for utility classes.
   */
  private HmmTrainer() {
    // nothing to do here really.
  }

  /**
   * Create an supervised initial estimate of an HMM Model based on a sequence
   * of observed and hidden states.
   *
   * @param nrOfHiddenStates The total number of hidden states
   * @param nrOfOutputStates The total number of output states
   * @param observedSequence Integer array containing the observed sequence
   * @param hiddenSequence   Integer array containing the hidden sequence
   * @param pseudoCount      Value that is assigned to non-occurring transitions to avoid zero
   *                         probabilities.
   * @return An initial model using the estimated parameters
   */
  public static HmmModel trainSupervised(int nrOfHiddenStates, int nrOfOutputStates, int[] observedSequence,
      int[] hiddenSequence, double pseudoCount) {
    // make sure the pseudo count is not zero
    pseudoCount = pseudoCount == 0 ? Double.MIN_VALUE : pseudoCount;

    // initialize the parameters
    DenseMatrix transitionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfHiddenStates);
    DenseMatrix emissionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfOutputStates);
    // assign a small initial probability that is larger than zero, so
    // unseen states will not get a zero probability
    transitionMatrix.assign(pseudoCount);
    emissionMatrix.assign(pseudoCount);
    // given no prior knowledge, we have to assume that all initial hidden
    // states are equally likely
    DenseVector initialProbabilities = new DenseVector(nrOfHiddenStates);
    initialProbabilities.assign(1.0 / nrOfHiddenStates);

    // now loop over the sequences to count the number of transitions
    countTransitions(transitionMatrix, emissionMatrix, observedSequence,
        hiddenSequence);

    // make sure that probabilities are normalized
    for (int i = 0; i < nrOfHiddenStates; i++) {
      // compute sum of probabilities for current row of transition matrix
      double sum = 0;
      for (int j = 0; j < nrOfHiddenStates; j++) {
        sum += transitionMatrix.getQuick(i, j);
      }
      // normalize current row of transition matrix
      for (int j = 0; j < nrOfHiddenStates; j++) {
        transitionMatrix.setQuick(i, j, transitionMatrix.getQuick(i, j) / sum);
      }
      // compute sum of probabilities for current row of emission matrix
      sum = 0;
      for (int j = 0; j < nrOfOutputStates; j++) {
        sum += emissionMatrix.getQuick(i, j);
      }
      // normalize current row of emission matrix
      for (int j = 0; j < nrOfOutputStates; j++) {
        emissionMatrix.setQuick(i, j, emissionMatrix.getQuick(i, j) / sum);
      }
    }

    // return a new model using the parameter estimations
    return new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);
  }

  /**
   * Function that counts the number of state->state and state->output
   * transitions for the given observed/hidden sequence.
   *
   * @param transitionMatrix transition matrix to use.
   * @param emissionMatrix emission matrix to use for counting.
   * @param observedSequence observation sequence to use.
   * @param hiddenSequence sequence of hidden states to use.
   */
  private static void countTransitions(Matrix transitionMatrix,
                                       Matrix emissionMatrix, int[] observedSequence, int[] hiddenSequence) {
    emissionMatrix.setQuick(hiddenSequence[0], observedSequence[0],
        emissionMatrix.getQuick(hiddenSequence[0], observedSequence[0]) + 1);
    for (int i = 1; i < observedSequence.length; ++i) {
      transitionMatrix
          .setQuick(hiddenSequence[i - 1], hiddenSequence[i], transitionMatrix
              .getQuick(hiddenSequence[i - 1], hiddenSequence[i]) + 1);
      emissionMatrix.setQuick(hiddenSequence[i], observedSequence[i],
          emissionMatrix.getQuick(hiddenSequence[i], observedSequence[i]) + 1);
    }
  }

  /**
   * Create an supervised initial estimate of an HMM Model based on a number of
   * sequences of observed and hidden states.
   *
   * @param nrOfHiddenStates The total number of hidden states
   * @param nrOfOutputStates The total number of output states
   * @param hiddenSequences Collection of hidden sequences to use for training
   * @param observedSequences Collection of observed sequences to use for training associated with hidden sequences.
   * @param pseudoCount      Value that is assigned to non-occurring transitions to avoid zero
   *                         probabilities.
   * @return An initial model using the estimated parameters
   */
  public static HmmModel trainSupervisedSequence(int nrOfHiddenStates,
                                                 int nrOfOutputStates, Collection<int[]> hiddenSequences,
                                                 Collection<int[]> observedSequences, double pseudoCount) {

    // make sure the pseudo count is not zero
    pseudoCount = pseudoCount == 0 ? Double.MIN_VALUE : pseudoCount;

    // initialize parameters
    DenseMatrix transitionMatrix = new DenseMatrix(nrOfHiddenStates,
        nrOfHiddenStates);
    DenseMatrix emissionMatrix = new DenseMatrix(nrOfHiddenStates,
        nrOfOutputStates);
    DenseVector initialProbabilities = new DenseVector(nrOfHiddenStates);

    // assign pseudo count to avoid zero probabilities
    transitionMatrix.assign(pseudoCount);
    emissionMatrix.assign(pseudoCount);
    initialProbabilities.assign(pseudoCount);

    // now loop over the sequences to count the number of transitions
    Iterator<int[]> hiddenSequenceIt = hiddenSequences.iterator();
    Iterator<int[]> observedSequenceIt = observedSequences.iterator();
    while (hiddenSequenceIt.hasNext() && observedSequenceIt.hasNext()) {
      // fetch the current set of sequences
      int[] hiddenSequence = hiddenSequenceIt.next();
      int[] observedSequence = observedSequenceIt.next();
      // increase the count for initial probabilities
      initialProbabilities.setQuick(hiddenSequence[0], initialProbabilities
          .getQuick(hiddenSequence[0]) + 1);
      countTransitions(transitionMatrix, emissionMatrix, observedSequence,
          hiddenSequence);
    }

    // make sure that probabilities are normalized
    double isum = 0; // sum of initial probabilities
    for (int i = 0; i < nrOfHiddenStates; i++) {
      isum += initialProbabilities.getQuick(i);
      // compute sum of probabilities for current row of transition matrix
      double sum = 0;
      for (int j = 0; j < nrOfHiddenStates; j++) {
        sum += transitionMatrix.getQuick(i, j);
      }
      // normalize current row of transition matrix
      for (int j = 0; j < nrOfHiddenStates; j++) {
        transitionMatrix.setQuick(i, j, transitionMatrix.getQuick(i, j) / sum);
      }
      // compute sum of probabilities for current row of emission matrix
      sum = 0;
      for (int j = 0; j < nrOfOutputStates; j++) {
        sum += emissionMatrix.getQuick(i, j);
      }
      // normalize current row of emission matrix
      for (int j = 0; j < nrOfOutputStates; j++) {
        emissionMatrix.setQuick(i, j, emissionMatrix.getQuick(i, j) / sum);
      }
    }
    // normalize the initial probabilities
    for (int i = 0; i < nrOfHiddenStates; ++i) {
      initialProbabilities.setQuick(i, initialProbabilities.getQuick(i) / isum);
    }

    // return a new model using the parameter estimates
    return new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);
  }

  /**
   * Iteratively train the parameters of the given initial model wrt to the
   * observed sequence using Viterbi training.
   *
   * @param initialModel     The initial model that gets iterated
   * @param observedSequence The sequence of observed states
   * @param pseudoCount      Value that is assigned to non-occurring transitions to avoid zero
   *                         probabilities.
   * @param epsilon          Convergence criteria
   * @param maxIterations    The maximum number of training iterations
   * @param scaled           Use Log-scaled implementation, this is computationally more
   *                         expensive but offers better numerical stability for large observed
   *                         sequences
   * @return The iterated model
   */
  public static HmmModel trainViterbi(HmmModel initialModel,
                                      int[] observedSequence, double pseudoCount, double epsilon,
                                      int maxIterations, boolean scaled) {

    // make sure the pseudo count is not zero
    pseudoCount = pseudoCount == 0 ? Double.MIN_VALUE : pseudoCount;

    // allocate space for iteration models
    HmmModel lastIteration = initialModel.clone();
    HmmModel iteration = initialModel.clone();

    // allocate space for Viterbi path calculation
    int[] viterbiPath = new int[observedSequence.length];
    int[][] phi = new int[observedSequence.length - 1][initialModel
        .getNrOfHiddenStates()];
    double[][] delta = new double[observedSequence.length][initialModel
        .getNrOfHiddenStates()];

    // now run the Viterbi training iteration
    for (int i = 0; i < maxIterations; ++i) {
      // compute the Viterbi path
      HmmAlgorithms.viterbiAlgorithm(viterbiPath, delta, phi, lastIteration,
          observedSequence, scaled);
      // Viterbi iteration uses the viterbi path to update
      // the probabilities
      Matrix emissionMatrix = iteration.getEmissionMatrix();
      Matrix transitionMatrix = iteration.getTransitionMatrix();

      // first, assign the pseudo count
      emissionMatrix.assign(pseudoCount);
      transitionMatrix.assign(pseudoCount);

      // now count the transitions
      countTransitions(transitionMatrix, emissionMatrix, observedSequence,
          viterbiPath);

      // and normalize the probabilities
      for (int j = 0; j < iteration.getNrOfHiddenStates(); ++j) {
        double sum = 0;
        // normalize the rows of the transition matrix
        for (int k = 0; k < iteration.getNrOfHiddenStates(); ++k) {
          sum += transitionMatrix.getQuick(j, k);
        }
        for (int k = 0; k < iteration.getNrOfHiddenStates(); ++k) {
          transitionMatrix
              .setQuick(j, k, transitionMatrix.getQuick(j, k) / sum);
        }
        // normalize the rows of the emission matrix
        sum = 0;
        for (int k = 0; k < iteration.getNrOfOutputStates(); ++k) {
          sum += emissionMatrix.getQuick(j, k);
        }
        for (int k = 0; k < iteration.getNrOfOutputStates(); ++k) {
          emissionMatrix.setQuick(j, k, emissionMatrix.getQuick(j, k) / sum);
        }
      }
      // check for convergence
      if (checkConvergence(lastIteration, iteration, epsilon)) {
        break;
      }
      // overwrite the last iterated model by the new iteration
      lastIteration.assign(iteration);
    }
    // we are done :)
    return iteration;
  }

  /**
   * Iteratively train the parameters of the given initial model wrt the
   * observed sequence using Baum-Welch training.
   *
   * @param initialModel     The initial model that gets iterated
   * @param observedSequence The sequence of observed states
   * @param epsilon          Convergence criteria
   * @param maxIterations    The maximum number of training iterations
   * @param scaled           Use log-scaled implementations of forward/backward algorithm. This
   *                         is computationally more expensive, but offers better numerical
   *                         stability for long output sequences.
   * @return The iterated model
   */
  public static HmmModel trainBaumWelch(HmmModel initialModel,
                                        int[] observedSequence, double epsilon, int maxIterations, boolean scaled) {
    // allocate space for the iterations
    HmmModel lastIteration = initialModel.clone();
    HmmModel iteration = initialModel.clone();

    // allocate space for baum-welch factors
    int hiddenCount = initialModel.getNrOfHiddenStates();
    int visibleCount = observedSequence.length;
    Matrix alpha = new DenseMatrix(visibleCount, hiddenCount);
    Matrix beta = new DenseMatrix(visibleCount, hiddenCount);

    // now run the baum Welch training iteration
    for (int it = 0; it < maxIterations; ++it) {
      // fetch emission and transition matrix of current iteration
      Vector initialProbabilities = iteration.getInitialProbabilities();
      Matrix emissionMatrix = iteration.getEmissionMatrix();
      Matrix transitionMatrix = iteration.getTransitionMatrix();

      // compute forward and backward factors
      HmmAlgorithms.forwardAlgorithm(alpha, iteration, observedSequence, scaled);
      HmmAlgorithms.backwardAlgorithm(beta, iteration, observedSequence, scaled);

      if (scaled) {
        logScaledBaumWelch(observedSequence, iteration, alpha, beta);
      } else {
        unscaledBaumWelch(observedSequence, iteration, alpha, beta);
      }
      // normalize transition/emission probabilities
      // and normalize the probabilities
      double isum = 0;
      for (int j = 0; j < iteration.getNrOfHiddenStates(); ++j) {
        double sum = 0;
        // normalize the rows of the transition matrix
        for (int k = 0; k < iteration.getNrOfHiddenStates(); ++k) {
          sum += transitionMatrix.getQuick(j, k);
        }
        for (int k = 0; k < iteration.getNrOfHiddenStates(); ++k) {
          transitionMatrix
              .setQuick(j, k, transitionMatrix.getQuick(j, k) / sum);
        }
        // normalize the rows of the emission matrix
        sum = 0;
        for (int k = 0; k < iteration.getNrOfOutputStates(); ++k) {
          sum += emissionMatrix.getQuick(j, k);
        }
        for (int k = 0; k < iteration.getNrOfOutputStates(); ++k) {
          emissionMatrix.setQuick(j, k, emissionMatrix.getQuick(j, k) / sum);
        }
        // normalization parameter for initial probabilities
        isum += initialProbabilities.getQuick(j);
      }
      // normalize initial probabilities
      for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
        initialProbabilities.setQuick(i, initialProbabilities.getQuick(i)
            / isum);
      }
      // check for convergence
      if (checkConvergence(lastIteration, iteration, epsilon)) {
        break;
      }
      // overwrite the last iterated model by the new iteration
      lastIteration.assign(iteration);
    }
    // we are done :)
    return iteration;
  }

  private static void unscaledBaumWelch(int[] observedSequence, HmmModel iteration, Matrix alpha, Matrix beta) {
    Vector initialProbabilities = iteration.getInitialProbabilities();
    Matrix emissionMatrix = iteration.getEmissionMatrix();
    Matrix transitionMatrix = iteration.getTransitionMatrix();
    double modelLikelihood = HmmEvaluator.modelLikelihood(alpha, false);

    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      initialProbabilities.setQuick(i, alpha.getQuick(0, i)
          * beta.getQuick(0, i));
    }

    // recompute transition probabilities
    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < iteration.getNrOfHiddenStates(); ++j) {
        double temp = 0;
        for (int t = 0; t < observedSequence.length - 1; ++t) {
          temp += alpha.getQuick(t, i)
              * emissionMatrix.getQuick(j, observedSequence[t + 1])
              * beta.getQuick(t + 1, j);
        }
        transitionMatrix.setQuick(i, j, transitionMatrix.getQuick(i, j)
            * temp / modelLikelihood);
      }
    }
    // recompute emission probabilities
    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < iteration.getNrOfOutputStates(); ++j) {
        double temp = 0;
        for (int t = 0; t < observedSequence.length; ++t) {
          // delta tensor
          if (observedSequence[t] == j) {
            temp += alpha.getQuick(t, i) * beta.getQuick(t, i);
          }
        }
        emissionMatrix.setQuick(i, j, temp / modelLikelihood);
      }
    }
  }

  private static void logScaledBaumWelch(int[] observedSequence, HmmModel iteration, Matrix alpha, Matrix beta) {
    Vector initialProbabilities = iteration.getInitialProbabilities();
    Matrix emissionMatrix = iteration.getEmissionMatrix();
    Matrix transitionMatrix = iteration.getTransitionMatrix();
    double modelLikelihood = HmmEvaluator.modelLikelihood(alpha, true);

    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      initialProbabilities.setQuick(i, Math.exp(alpha.getQuick(0, i) + beta.getQuick(0, i)));
    }

    // recompute transition probabilities
    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < iteration.getNrOfHiddenStates(); ++j) {
        double sum = Double.NEGATIVE_INFINITY; // log(0)
        for (int t = 0; t < observedSequence.length - 1; ++t) {
          double temp = alpha.getQuick(t, i)
              + Math.log(emissionMatrix.getQuick(j, observedSequence[t + 1]))
              + beta.getQuick(t + 1, j);
          if (temp > Double.NEGATIVE_INFINITY) {
            // handle 0-probabilities
            sum = temp + Math.log1p(Math.exp(sum - temp));
          }
        }
        transitionMatrix.setQuick(i, j, transitionMatrix.getQuick(i, j)
            * Math.exp(sum - modelLikelihood));
      }
    }
    // recompute emission probabilities
    for (int i = 0; i < iteration.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < iteration.getNrOfOutputStates(); ++j) {
        double sum = Double.NEGATIVE_INFINITY; // log(0)
        for (int t = 0; t < observedSequence.length; ++t) {
          // delta tensor
          if (observedSequence[t] == j) {
            double temp = alpha.getQuick(t, i) + beta.getQuick(t, i);
            if (temp > Double.NEGATIVE_INFINITY) {
              // handle 0-probabilities
              sum = temp + Math.log1p(Math.exp(sum - temp));
            }
          }
        }
        emissionMatrix.setQuick(i, j, Math.exp(sum - modelLikelihood));
      }
    }
  }

  /**
   * Check convergence of two HMM models by computing a simple distance between
   * emission / transition matrices
   *
   * @param oldModel Old HMM Model
   * @param newModel New HMM Model
   * @param epsilon  Convergence Factor
   * @return true if training converged to a stable state.
   */
  private static boolean checkConvergence(HmmModel oldModel, HmmModel newModel,
                                          double epsilon) {
    // check convergence of transitionProbabilities
    Matrix oldTransitionMatrix = oldModel.getTransitionMatrix();
    Matrix newTransitionMatrix = newModel.getTransitionMatrix();
    double diff = 0;
    for (int i = 0; i < oldModel.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < oldModel.getNrOfHiddenStates(); ++j) {
        double tmp = oldTransitionMatrix.getQuick(i, j)
            - newTransitionMatrix.getQuick(i, j);
        diff += tmp * tmp;
      }
    }
    double norm = Math.sqrt(diff);
    diff = 0;
    // check convergence of emissionProbabilities
    Matrix oldEmissionMatrix = oldModel.getEmissionMatrix();
    Matrix newEmissionMatrix = newModel.getEmissionMatrix();
    for (int i = 0; i < oldModel.getNrOfHiddenStates(); i++) {
      for (int j = 0; j < oldModel.getNrOfOutputStates(); j++) {

        double tmp = oldEmissionMatrix.getQuick(i, j)
            - newEmissionMatrix.getQuick(i, j);
        diff += tmp * tmp;
      }
    }
    norm += Math.sqrt(diff);
    // iteration has converged :)
    return norm < epsilon;
  }

}
