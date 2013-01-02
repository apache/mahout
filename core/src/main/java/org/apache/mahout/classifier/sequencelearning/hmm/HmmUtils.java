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
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;

import com.google.common.base.Preconditions;

/**
 * A collection of utilities for handling HMMModel objects.
 */
public final class HmmUtils {

  /**
   * No public constructor for utility classes.
   */
  private HmmUtils() {
    // nothing to do here really.
  }

  /**
   * Compute the cumulative transition probability matrix for the given HMM
   * model. Matrix where each row i is the cumulative distribution of the
   * transition probability distribution for hidden state i.
   *
   * @param model The HMM model for which the cumulative transition matrix should be
   *              computed
   * @return The computed cumulative transition matrix.
   */
  public static Matrix getCumulativeTransitionMatrix(HmmModel model) {
    // fetch the needed parameters from the model
    int hiddenStates = model.getNrOfHiddenStates();
    Matrix transitionMatrix = model.getTransitionMatrix();
    // now compute the cumulative transition matrix
    Matrix resultMatrix = new DenseMatrix(hiddenStates, hiddenStates);
    for (int i = 0; i < hiddenStates; ++i) {
      double sum = 0;
      for (int j = 0; j < hiddenStates; ++j) {
        sum += transitionMatrix.get(i, j);
        resultMatrix.set(i, j, sum);
      }
      resultMatrix.set(i, hiddenStates - 1, 1.0);
      // make sure the last
      // state has always a
      // cumulative
      // probability of
      // exactly 1.0
    }
    return resultMatrix;
  }

  /**
   * Compute the cumulative output probability matrix for the given HMM model.
   * Matrix where each row i is the cumulative distribution of the output
   * probability distribution for hidden state i.
   *
   * @param model The HMM model for which the cumulative output matrix should be
   *              computed
   * @return The computed cumulative output matrix.
   */
  public static Matrix getCumulativeOutputMatrix(HmmModel model) {
    // fetch the needed parameters from the model
    int hiddenStates = model.getNrOfHiddenStates();
    int outputStates = model.getNrOfOutputStates();
    Matrix outputMatrix = model.getEmissionMatrix();
    // now compute the cumulative output matrix
    Matrix resultMatrix = new DenseMatrix(hiddenStates, outputStates);
    for (int i = 0; i < hiddenStates; ++i) {
      double sum = 0;
      for (int j = 0; j < outputStates; ++j) {
        sum += outputMatrix.get(i, j);
        resultMatrix.set(i, j, sum);
      }
      resultMatrix.set(i, outputStates - 1, 1.0);
      // make sure the last
      // output state has
      // always a cumulative
      // probability of 1.0
    }
    return resultMatrix;
  }

  /**
   * Compute the cumulative distribution of the initial hidden state
   * probabilities for the given HMM model.
   *
   * @param model The HMM model for which the cumulative initial state probabilities
   *              should be computed
   * @return The computed cumulative initial state probability vector.
   */
  public static Vector getCumulativeInitialProbabilities(HmmModel model) {
    // fetch the needed parameters from the model
    int hiddenStates = model.getNrOfHiddenStates();
    Vector initialProbabilities = model.getInitialProbabilities();
    // now compute the cumulative output matrix
    Vector resultVector = new DenseVector(initialProbabilities.size());
    double sum = 0;
    for (int i = 0; i < hiddenStates; ++i) {
      sum += initialProbabilities.get(i);
      resultVector.set(i, sum);
    }
    resultVector.set(hiddenStates - 1, 1.0); // make sure the last initial
    // hidden state probability
    // has always a cumulative
    // probability of 1.0
    return resultVector;
  }

  /**
   * Validates an HMM model set
   *
   * @param model model to sanity check.
   */
  public static void validate(HmmModel model) {
    if (model == null) {
      return; // empty models are valid
    }

    /*
     * The number of hidden states is positive.
     */
    Preconditions.checkArgument(model.getNrOfHiddenStates() > 0,
      "Error: The number of hidden states has to be greater than 0");
    
    /*
     * The number of output states is positive.
     */
    Preconditions.checkArgument(model.getNrOfOutputStates() > 0,
      "Error: The number of output states has to be greater than 0!");

    /*
     * The size of the vector of initial probabilities is equal to the number of
     * the hidden states. Each initial probability is non-negative. The sum of
     * initial probabilities is equal to 1.
     */
    Preconditions.checkArgument(model.getInitialProbabilities() != null
      && model.getInitialProbabilities().size() == model.getNrOfHiddenStates(),
      "Error: The vector of initial probabilities is not initialized!");
    
    double sum = 0;
    for (int i = 0; i < model.getInitialProbabilities().size(); i++) {
      Preconditions.checkArgument(model.getInitialProbabilities().get(i) >= 0,
        "Error: Initial probability of state %d is negative", i);
      sum += model.getInitialProbabilities().get(i);
    }
    Preconditions.checkArgument(Math.abs(sum - 1) <= 0.00001,
                                "Error: Initial probabilities do not add up to 1");
    /*
     * The row size of the output matrix is equal to the number of the hidden
     * states. The column size is equal to the number of output states. Each
     * probability of the matrix is non-negative. The sum of each row is equal
     * to 1.
     */
    Preconditions.checkNotNull(model.getEmissionMatrix(), "Error: The output state matrix is not initialized!");
    Preconditions.checkArgument(model.getEmissionMatrix().numRows() == model.getNrOfHiddenStates()
        && model.getEmissionMatrix().numCols() == model.getNrOfOutputStates(),
        "Error: The output state matrix is not of the form nrOfHiddenStates x nrOfOutputStates");
    for (int i = 0; i < model.getEmissionMatrix().numRows(); i++) {
      sum = 0;
      for (int j = 0; j < model.getEmissionMatrix().numCols(); j++) {
        Preconditions.checkArgument(model.getEmissionMatrix().get(i, j) >= 0,
            "The output state probability from hidden state " + i + " to output state " + j + " is negative");
        sum += model.getEmissionMatrix().get(i, j);
      }
      Preconditions.checkArgument(Math.abs(sum - 1) <= 0.00001,
        "Error: The output state probabilities for hidden state %d don't add up to 1", i);
    }

    /*
     * The size of both dimension of the transition matrix is equal to the
     * number of the hidden states. Each probability of the matrix is
     * non-negative. The sum of each row in transition matrix is equal to 1.
     */
    Preconditions.checkArgument(model.getTransitionMatrix() != null,
      "Error: The hidden state matrix is not initialized!");
    Preconditions.checkArgument(model.getTransitionMatrix().numRows() == model.getNrOfHiddenStates()
      && model.getTransitionMatrix().numCols() == model.getNrOfHiddenStates(),
      "Error: The output state matrix is not of the form nrOfHiddenStates x nrOfHiddenStates");
    for (int i = 0; i < model.getTransitionMatrix().numRows(); i++) {
      sum = 0;
      for (int j = 0; j < model.getTransitionMatrix().numCols(); j++) {
        Preconditions.checkArgument(model.getTransitionMatrix().get(i, j) >= 0,
          "Error: The transition probability from hidden state %d to hidden state %d is negative", i, j);
        sum += model.getTransitionMatrix().get(i, j);
      }
      Preconditions.checkArgument(Math.abs(sum - 1) <= 0.00001,
        "Error: The transition probabilities for hidden state " + i + " don't add up to 1.");
    }
  }

  /**
   * Encodes a given collection of state names by the corresponding state IDs
   * registered in a given model.
   *
   * @param model        Model to provide the encoding for
   * @param sequence     Collection of state names
   * @param observed     If set, the sequence is encoded as a sequence of observed states,
   *                     else it is encoded as sequence of hidden states
   * @param defaultValue The default value in case a state is not known
   * @return integer array containing the encoded state IDs
   */
  public static int[] encodeStateSequence(HmmModel model,
                                          Collection<String> sequence, boolean observed, int defaultValue) {
    int[] encoded = new int[sequence.size()];
    Iterator<String> seqIter = sequence.iterator();
    for (int i = 0; i < sequence.size(); ++i) {
      String nextState = seqIter.next();
      int nextID;
      if (observed) {
        nextID = model.getOutputStateID(nextState);
      } else {
        nextID = model.getHiddenStateID(nextState);
      }
      // if the ID is -1, use the default value
      encoded[i] = nextID < 0 ? defaultValue : nextID;
    }
    return encoded;
  }

  /**
   * Decodes a given collection of state IDs into the corresponding state names
   * registered in a given model.
   *
   * @param model        model to use for retrieving state names
   * @param sequence     int array of state IDs
   * @param observed     If set, the sequence is encoded as a sequence of observed states,
   *                     else it is encoded as sequence of hidden states
   * @param defaultValue The default value in case a state is not known
   * @return list containing the decoded state names
   */
  public static List<String> decodeStateSequence(HmmModel model,
                                                 int[] sequence,
                                                 boolean observed,
                                                 String defaultValue) {
    List<String> decoded = Lists.newArrayListWithCapacity(sequence.length);
    for (int position : sequence) {
      String nextState;
      if (observed) {
        nextState = model.getOutputStateName(position);
      } else {
        nextState = model.getHiddenStateName(position);
      }
      // if null was returned, use the default value
      decoded.add(nextState == null ? defaultValue : nextState);
    }
    return decoded;
  }

  /**
   * Function used to normalize the probabilities of a given HMM model
   *
   * @param model model to normalize
   */
  public static void normalizeModel(HmmModel model) {
    Vector ip = model.getInitialProbabilities();
    Matrix emission = model.getEmissionMatrix();
    Matrix transition = model.getTransitionMatrix();
    // check normalization for all probabilities
    double isum = 0;
    for (int i = 0; i < model.getNrOfHiddenStates(); ++i) {
      isum += ip.getQuick(i);
      double sum = 0;
      for (int j = 0; j < model.getNrOfHiddenStates(); ++j) {
        sum += transition.getQuick(i, j);
      }
      if (sum != 1.0) {
        for (int j = 0; j < model.getNrOfHiddenStates(); ++j) {
          transition.setQuick(i, j, transition.getQuick(i, j) / sum);
        }
      }
      sum = 0;
      for (int j = 0; j < model.getNrOfOutputStates(); ++j) {
        sum += emission.getQuick(i, j);
      }
      if (sum != 1.0) {
        for (int j = 0; j < model.getNrOfOutputStates(); ++j) {
          emission.setQuick(i, j, emission.getQuick(i, j) / sum);
        }
      }
    }
    if (isum != 1.0) {
      for (int i = 0; i < model.getNrOfHiddenStates(); ++i) {
        ip.setQuick(i, ip.getQuick(i) / isum);
      }
    }
  }

  /**
   * Method to reduce the size of an HMMmodel by converting the models
   * DenseMatrix/DenseVectors to sparse implementations and setting every value
   * < threshold to 0
   *
   * @param model model to truncate
   * @param threshold minimum value a model entry must have to be retained.
   * @return Truncated model
   */
  public static HmmModel truncateModel(HmmModel model, double threshold) {
    Vector ip = model.getInitialProbabilities();
    Matrix em = model.getEmissionMatrix();
    Matrix tr = model.getTransitionMatrix();
    // allocate the sparse data structures
    RandomAccessSparseVector sparseIp = new RandomAccessSparseVector(model
        .getNrOfHiddenStates());
    SparseMatrix sparseEm = new SparseMatrix(model.getNrOfHiddenStates(), model.getNrOfOutputStates());
    SparseMatrix sparseTr = new SparseMatrix(model.getNrOfHiddenStates(), model.getNrOfHiddenStates());
    // now transfer the values
    for (int i = 0; i < model.getNrOfHiddenStates(); ++i) {
      double value = ip.getQuick(i);
      if (value > threshold) {
        sparseIp.setQuick(i, value);
      }
      for (int j = 0; j < model.getNrOfHiddenStates(); ++j) {
        value = tr.getQuick(i, j);
        if (value > threshold) {
          sparseTr.setQuick(i, j, value);
        }
      }

      for (int j = 0; j < model.getNrOfOutputStates(); ++j) {
        value = em.getQuick(i, j);
        if (value > threshold) {
          sparseEm.setQuick(i, j, value);
        }
      }
    }
    // create a new model
    HmmModel sparseModel = new HmmModel(sparseTr, sparseEm, sparseIp);
    // normalize the model
    normalizeModel(sparseModel);
    // register the names
    sparseModel.registerHiddenStateNames(model.getHiddenStateNames());
    sparseModel.registerOutputStateNames(model.getOutputStateNames());
    // and return
    return sparseModel;
  }
}
