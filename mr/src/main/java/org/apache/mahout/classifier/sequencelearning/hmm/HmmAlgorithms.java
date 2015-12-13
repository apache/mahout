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

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Class containing implementations of the three major HMM algorithms: forward,
 * backward and Viterbi
 */
public final class HmmAlgorithms {

    public enum ScalingMethod {
	NOSCALING,
	    LOGSCALING,
	    RESCALING
	    }

  /**
   * No public constructors for utility classes.
   */
  private HmmAlgorithms() {
    // nothing to do here really
  }

  /**
   * External function to compute a matrix of alpha factors
   *
   * @param model        model to run forward algorithm for.
   * @param observations observation sequence to train on.
   * @param scaling       Scaling method to use (no scaling, log scaling or rescaling)
   * @return matrix of alpha factors.
   */
    public static Matrix forwardAlgorithm(HmmModel model, int[] observations, ScalingMethod scaling, double[] scalingFactors) {
    Matrix alpha = new DenseMatrix(observations.length, model.getNrOfHiddenStates());
    forwardAlgorithm(alpha, model, observations, scaling, scalingFactors);

    return alpha;
  }

  /**
   * Internal function to compute the alpha factors
   *
   * @param alpha        matrix to store alpha factors in.
   * @param model        model to use for alpha factor computation.
   * @param observations observation sequence seen.
   * @param scaled       set to true if log-scaled beta factors should be computed.
   */
    static void forwardAlgorithm(Matrix alpha, HmmModel model, int[] observations, ScalingMethod scaling, double[] scalingFactors) {

    // fetch references to the model parameters
    Vector ip = model.getInitialProbabilities();
    Matrix b = model.getEmissionMatrix();
    Matrix a = model.getTransitionMatrix();

    if (scaling == ScalingMethod.LOGSCALING) { // compute log scaled alpha values
      // Initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        alpha.setQuick(0, i, Math.log(ip.getQuick(i) * b.getQuick(i, observations[0])));
      }

      // Induction
      for (int t = 1; t < observations.length; t++) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double sum = Double.NEGATIVE_INFINITY; // log(0)
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
            double tmp = alpha.getQuick(t - 1, j) + Math.log(a.getQuick(j, i));
            if (tmp > Double.NEGATIVE_INFINITY) {
              // make sure we handle log(0) correctly
              sum = tmp + Math.log1p(Math.exp(sum - tmp));
            }
          }
          alpha.setQuick(t, i, sum + Math.log(b.getQuick(i, observations[t])));
        }
      }
    } else if (scaling == ScalingMethod.RESCALING) {
	Matrix alphaTemp = new DenseMatrix(observations.length, model.getNrOfHiddenStates());

      // Initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        alphaTemp.setQuick(0, i, ip.getQuick(i) * b.getQuick(i, observations[0]));
      }

      double sum = 0.0;
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
	  sum += alphaTemp.getQuick(0, i);
      }

      scalingFactors[0] = 1.0/sum;

      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
	  alpha.setQuick(0, i, alphaTemp.getQuick(0, i) * scalingFactors[0]);
      }

      // Induction
      for (int t = 1; t < observations.length; t++) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double suma = 0.0;
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
			  suma += alpha.getQuick(t - 1, j) * a.getQuick(j, i) * b.getQuick(i, observations[t]);
          }
		  alphaTemp.setQuick(t, i, suma);
        }
		double sumt = 0.0;
		for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
			sumt += alphaTemp.getQuick(t, i);
		}
		scalingFactors[t] = 1.0/sumt;
		for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
		    alpha.setQuick(t, i, scalingFactors[t] * alphaTemp.getQuick(t, i));
		}
      }
    } else {
      // Initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        alpha.setQuick(0, i, ip.getQuick(i) * b.getQuick(i, observations[0]));
      }

      // Induction
      for (int t = 1; t < observations.length; t++) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double sum = 0.0;
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
            sum += alpha.getQuick(t - 1, j) * a.getQuick(j, i);
          }
          alpha.setQuick(t, i, sum * b.getQuick(i, observations[t]));
        }
      }
    }
  }

  /**
   * External function to compute a matrix of beta factors
   *
   * @param model        model to use for estimation.
   * @param observations observation sequence seen.
   * @param scaled       Set to true if log-scaled beta factors should be computed.
   * @return beta factors based on the model and observation sequence.
   */
  public static Matrix backwardAlgorithm(HmmModel model, int[] observations, ScalingMethod scaling, double[] scalingFactors) {
    // initialize the matrix
    Matrix beta = new DenseMatrix(observations.length, model.getNrOfHiddenStates());
    // compute the beta factors
    backwardAlgorithm(beta, model, observations, scaling, scalingFactors);

    return beta;
  }

  /**
   * Internal function to compute the beta factors
   *
   * @param beta         Matrix to store resulting factors in.
   * @param model        model to use for factor estimation.
   * @param observations sequence of observations to estimate.
   * @param scaled       set to true to compute log-scaled parameters.
   */
  static void backwardAlgorithm(Matrix beta, HmmModel model, int[] observations, ScalingMethod scaling, double[] scalingFactors) {
    // fetch references to the model parameters
    Matrix b = model.getEmissionMatrix();
    Matrix a = model.getTransitionMatrix();

    if (scaling == ScalingMethod.LOGSCALING) { // compute log-scaled factors
      // initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        beta.setQuick(observations.length - 1, i, 0);
      }

      // induction
      for (int t = observations.length - 2; t >= 0; t--) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double sum = Double.NEGATIVE_INFINITY; // log(0)
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
            double tmp = beta.getQuick(t + 1, j) + Math.log(a.getQuick(i, j))
                + Math.log(b.getQuick(j, observations[t + 1]));
            if (tmp > Double.NEGATIVE_INFINITY) {
              // handle log(0)
              sum = tmp + Math.log1p(Math.exp(sum - tmp));
            }
          }
          beta.setQuick(t, i, sum);
        }
      }
    } else if (scaling == ScalingMethod.RESCALING) {
		Matrix betaTemp = new DenseMatrix(observations.length, model.getNrOfHiddenStates());
		// initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        betaTemp.setQuick(observations.length - 1, i, 1);
		beta.setQuick(observations.length - 1, i, scalingFactors[observations.length - 1] * betaTemp.getQuick(observations.length - 1, i));
      }
      // induction
      for (int t = observations.length - 2; t >= 0; t--) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double sum = 0;
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
            sum += beta.getQuick(t + 1, j) * a.getQuick(i, j) * b.getQuick(j, observations[t + 1]);
          }
          betaTemp.setQuick(t, i, sum);
          beta.setQuick(t, i, betaTemp.getQuick(t, i) * scalingFactors[t]);		  
        }
      }
    } else {
      // initialization
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        beta.setQuick(observations.length - 1, i, 1);
      }
      // induction
      for (int t = observations.length - 2; t >= 0; t--) {
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          double sum = 0;
          for (int j = 0; j < model.getNrOfHiddenStates(); j++) {
            sum += beta.getQuick(t + 1, j) * a.getQuick(i, j) * b.getQuick(j, observations[t + 1]);
          }
          beta.setQuick(t, i, sum);
        }
      }
    }
  }

  /**
   * Viterbi algorithm to compute the most likely hidden sequence for a given
   * model and observed sequence
   *
   * @param model        HmmModel for which the Viterbi path should be computed
   * @param observations Sequence of observations
   * @param scaled       Use log-scaled computations, this requires higher computational
   *                     effort but is numerically more stable for large observation
   *                     sequences
   * @return nrOfObservations 1D int array containing the most likely hidden
   *         sequence
   */
  public static int[] viterbiAlgorithm(HmmModel model, int[] observations, boolean scaled) {

    // probability that the most probable hidden states ends at state i at
    // time t
    double[][] delta = new double[observations.length][model
        .getNrOfHiddenStates()];

    // previous hidden state in the most probable state leading up to state
    // i at time t
    int[][] phi = new int[observations.length - 1][model.getNrOfHiddenStates()];

    // initialize the return array
    int[] sequence = new int[observations.length];

    viterbiAlgorithm(sequence, delta, phi, model, observations, scaled);

    return sequence;
  }

  /**
   * Internal version of the viterbi algorithm, allowing to reuse existing
   * arrays instead of allocating new ones
   *
   * @param sequence     NrOfObservations 1D int array for storing the viterbi sequence
   * @param delta        NrOfObservations x NrHiddenStates 2D double array for storing the
   *                     delta factors
   * @param phi          NrOfObservations-1 x NrHiddenStates 2D int array for storing the
   *                     phi values
   * @param model        HmmModel for which the viterbi path should be computed
   * @param observations Sequence of observations
   * @param scaled       Use log-scaled computations, this requires higher computational
   *                     effort but is numerically more stable for large observation
   *                     sequences
   */
  static void viterbiAlgorithm(int[] sequence, double[][] delta, int[][] phi, HmmModel model, int[] observations,
      boolean scaled) {
    // fetch references to the model parameters
    Vector ip = model.getInitialProbabilities();
    Matrix b = model.getEmissionMatrix();
    Matrix a = model.getTransitionMatrix();

    // Initialization
    if (scaled) {
      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        delta[0][i] = Math.log(ip.getQuick(i) * b.getQuick(i, observations[0]));
      }
    } else {

      for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
        delta[0][i] = ip.getQuick(i) * b.getQuick(i, observations[0]);
      }
    }

    // Induction
    // iterate over the time
    if (scaled) {
      for (int t = 1; t < observations.length; t++) {
        // iterate over the hidden states
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          int maxState = 0;
          double maxProb = delta[t - 1][0] + Math.log(a.getQuick(0, i));
          for (int j = 1; j < model.getNrOfHiddenStates(); j++) {
            double prob = delta[t - 1][j] + Math.log(a.getQuick(j, i));
            if (prob > maxProb) {
              maxProb = prob;
              maxState = j;
            }
          }
          delta[t][i] = maxProb + Math.log(b.getQuick(i, observations[t]));
          phi[t - 1][i] = maxState;
        }
      }
    } else {
      for (int t = 1; t < observations.length; t++) {
        // iterate over the hidden states
        for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
          // find the maximum probability and most likely state
          // leading up
          // to this
          int maxState = 0;
          double maxProb = delta[t - 1][0] * a.getQuick(0, i);
          for (int j = 1; j < model.getNrOfHiddenStates(); j++) {
            double prob = delta[t - 1][j] * a.getQuick(j, i);
            if (prob > maxProb) {
              maxProb = prob;
              maxState = j;
            }
          }
          delta[t][i] = maxProb * b.getQuick(i, observations[t]);
          phi[t - 1][i] = maxState;
        }
      }
    }

    // find the most likely end state for initialization
    double maxProb;
    if (scaled) {
      maxProb = Double.NEGATIVE_INFINITY;
    } else {
      maxProb = 0.0;
    }
    for (int i = 0; i < model.getNrOfHiddenStates(); i++) {
      if (delta[observations.length - 1][i] > maxProb) {
        maxProb = delta[observations.length - 1][i];
        sequence[observations.length - 1] = i;
      }
    }

    // now backtrack to find the most likely hidden sequence
    for (int t = observations.length - 2; t >= 0; t--) {
      sequence[t] = phi[t][sequence[t + 1]];
    }
  }

}
