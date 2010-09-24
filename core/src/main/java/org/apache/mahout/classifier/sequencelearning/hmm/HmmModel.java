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

import java.lang.reflect.Type;
import java.util.Map;
import java.util.Random;

import org.apache.commons.collections.BidiMap;
import org.apache.commons.collections.bidimap.TreeBidiMap;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.JsonMatrixAdapter;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;

/**
 * Main class defining a Hidden Markov Model
 *
 * @author mheimel
 */
public class HmmModel implements JsonDeserializer<HmmModel>,
    JsonSerializer<HmmModel>, Cloneable {

  /**
   * No-args constructed needed for json de-serialization
   */
  private HmmModel() {
    // do nothing
  }

  /**
   * Get a copy of this model
   */
  public HmmModel clone() throws CloneNotSupportedException {
    super.clone();
    HmmModel model = new HmmModel(transitionMatrix.clone(), emissionMatrix
        .clone(), initialProbabilities.clone());
    if (hiddenStateNames != null)
      model.hiddenStateNames = new TreeBidiMap(hiddenStateNames);
    if (outputStateNames != null)
      model.outputStateNames = new TreeBidiMap(outputStateNames);
    return model;
  }

  /**
   * Assign the content of another HMM model to this one
   *
   * @param model The HmmModel that will be assigned to this one
   */
  public void assign(HmmModel model) {
    this.nrOfHiddenStates = model.nrOfHiddenStates;
    this.nrOfOutputStates = model.nrOfOutputStates;
    this.hiddenStateNames = model.hiddenStateNames;
    this.outputStateNames = model.outputStateNames;
    // for now clone the matrix/vectors
    this.initialProbabilities = model.initialProbabilities.clone();
    this.emissionMatrix = model.emissionMatrix.clone();
    this.transitionMatrix = model.transitionMatrix.clone();
  }

  /**
   * Construct a valid random Hidden-Markov parameter set with the given number
   * of hidden and output states using a given seed.
   *
   * @param nrOfHiddenStates Number of hidden states
   * @param nrOfOutputStates Number of output states
   * @param seed             Seed for the random initialization, if set to 0 the current time
   *                         is used
   */
  public HmmModel(int nrOfHiddenStates, int nrOfOutputStates, long seed) {
    this.nrOfHiddenStates = nrOfHiddenStates;
    this.nrOfOutputStates = nrOfOutputStates;
    this.transitionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfHiddenStates);
    this.emissionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfOutputStates);
    this.initialProbabilities = new DenseVector(nrOfHiddenStates);
    // initialize a random, valid parameter set
    initRandomParameters(seed);
  }

  /**
   * Construct a valid random Hidden-Markov parameter set with the given number
   * of hidden and output states.
   *
   * @param nrOfHiddenStates Number of hidden states
   * @param nrOfOutputStates Number of output states
   */
  public HmmModel(int nrOfHiddenStates, int nrOfOutputStates) {
    this(nrOfHiddenStates, nrOfOutputStates, 0);
  }

  /**
   * Generates a Hidden Markov model using the specified parameters
   *
   * @param transitionMatrix     transition probabilities.
   * @param emissionMatrix       emission probabilities.
   * @param initialProbabilities initial start probabilities.
   * @throws IllegalArgumentException If the given parameter set is invalid
   */
  public HmmModel(Matrix transitionMatrix, Matrix emissionMatrix,
                  Vector initialProbabilities) {
    this.nrOfHiddenStates = initialProbabilities.size();
    this.nrOfOutputStates = emissionMatrix.numCols();
    this.transitionMatrix = transitionMatrix;
    this.emissionMatrix = emissionMatrix;
    this.initialProbabilities = initialProbabilities;
  }

  /**
   * Initialize a valid random set of HMM parameters
   *
   * @param seed seed to use for Random initialization. Use 0 to use Java-built-in-version.
   */
  private void initRandomParameters(long seed) {
    Random rand;
    // initialize the random number generator
    if (seed == 0)
      rand = RandomUtils.getRandom();
    else
      rand = RandomUtils.getRandom(seed);
    // initialize the initial Probabilities
    double sum = 0; // used for normalization
    for (int i = 0; i < nrOfHiddenStates; i++) {
      double nextRand = rand.nextDouble();
      initialProbabilities.set(i, nextRand);
      sum += nextRand;
    }
    // "normalize" the vector to generate probabilities
    initialProbabilities = initialProbabilities.divide(sum);

    // initialize the transition matrix
    double[] values = new double[nrOfHiddenStates];
    for (int i = 0; i < nrOfHiddenStates; i++) {
      sum = 0;
      for (int j = 0; j < nrOfHiddenStates; j++) {
        values[j] = rand.nextDouble();
        sum += values[j];
      }
      // normalize the random values to obtain probabilities
      for (int j = 0; j < nrOfHiddenStates; j++)
        values[j] /= sum;
      // set this row of the transition matrix
      transitionMatrix.set(i, values);
    }

    // initialize the output matrix
    values = new double[nrOfOutputStates];
    for (int i = 0; i < nrOfHiddenStates; i++) {
      sum = 0;
      for (int j = 0; j < nrOfOutputStates; j++) {
        values[j] = rand.nextDouble();
        sum += values[j];
      }
      // normalize the random values to obtain probabilities
      for (int j = 0; j < nrOfOutputStates; j++)
        values[j] /= sum;
      // set this row of the output matrix
      emissionMatrix.set(i, values);
    }
  }

  /**
   * Number of hidden states
   */
  private int nrOfHiddenStates;

  /**
   * Getter Method for the number of hidden states
   *
   * @return Number of hidden states
   */
  public int getNrOfHiddenStates() {
    return nrOfHiddenStates;
  }

  /**
   * Number of output states
   */
  private int nrOfOutputStates;

  /**
   * Getter Method for the number of output states
   *
   * @return Number of output states
   */
  public int getNrOfOutputStates() {
    return nrOfOutputStates;
  }

  /**
   * Transition matrix containing the transition probabilities between hidden
   * states. TransitionMatrix(i,j) is the probability that we change from hidden
   * state i to hidden state j In general: P(h(t+1)=h_j | h(t) = h_i) =
   * transitionMatrix(i,j) Since we have to make sure that each hidden state can
   * be "left", the following normalization condition has to hold:
   * sum(transitionMatrix(i,j),j=1..hiddenStates) = 1
   */
  private Matrix transitionMatrix;

  /**
   * Getter function to get the hidden state transition matrix
   *
   * @return returns the model's transition matrix.
   */
  public Matrix getTransitionMatrix() {
    return transitionMatrix;
  }

  /**
   * Output matrix containing the probabilities that we observe a given output
   * state given a hidden state. outputMatrix(i,j) is the probability that we
   * observe output state j if we are in hidden state i Formally: P(o(t)=o_j |
   * h(t)=h_i) = outputMatrix(i,j) Since we always have an observation for each
   * hidden state, the following normalization condition has to hold:
   * sum(outputMatrix(i,j),j=1..outputStates) = 1
   */
  private Matrix emissionMatrix;

  /**
   * Getter function to get the output state probability matrix
   *
   * @return returns the models emission matrix.
   */
  public Matrix getEmissionMatrix() {
    return emissionMatrix;
  }

  /**
   * Vector containing the initial hidden state probabilities. That is
   * P(h(0)=h_i) = initialProbabilities(i). Since we are dealing with
   * probabilities the following normalization condition has to hold:
   * sum(initialProbabilities(i),i=1..hiddenStates) = 1
   */
  private Vector initialProbabilities;

  /**
   * Getter function to return the vector of initial hidden state probabilities
   *
   * @return returns the model's init probabilities.
   */
  public Vector getInitialProbabilities() {
    return initialProbabilities;
  }

  /**
   * Bi-Directional Map for storing the hidden state names
   */
  private BidiMap hiddenStateNames;

  /**
   * Getter method for the hidden state Names map
   *
   * @return hidden state names.
   */
  @SuppressWarnings("unchecked")
  public Map<String, Integer> getHiddenStateNames() {
    return hiddenStateNames;
  }

  /**
   * Register an array of hidden state Names. We assume that the state name at
   * position i has the ID i
   *
   * @param stateNames names of hidden states.
   */
  public void registerHiddenStateNames(String[] stateNames) {
    if (stateNames != null) {
      hiddenStateNames = new TreeBidiMap();
      for (int i = 0; i < stateNames.length; ++i) {
        hiddenStateNames.put(stateNames[i], i);
      }
    }
  }

  /**
   * Register a map of hidden state Names/state IDs
   *
   * @param stateNames <String,Integer> Map that assigns each state name an integer ID
   */
  public void registerHiddenStateNames(Map<String, Integer> stateNames) {
    if (stateNames != null)
      hiddenStateNames = new TreeBidiMap(stateNames);
  }

  /**
   * Lookup the name for the given hidden state ID
   *
   * @param id Integer id of the hidden state
   * @return String containing the name for the given ID, null if this ID is not
   *         known or no hidden state names were specified
   */
  public String getHiddenStateName(int id) {
    if (hiddenStateNames == null)
      return null;
    return (String) hiddenStateNames.getKey(id);
  }

  /**
   * Lookup the ID for the given hidden state name
   *
   * @param name Name of the hidden state
   * @return int containing the ID for the given name, -1 if this name is not
   *         known or no hidden state names were specified
   */
  public int getHiddenStateID(String name) {
    if (hiddenStateNames == null)
      return -1;
    Integer tmp = (Integer) hiddenStateNames.get(name);
    return (tmp == null) ? -1 : tmp;
  }

  /**
   * Bi-directional Map for storing the observed state names
   */
  private BidiMap outputStateNames;

  /**
   * Getter method for the output state Names map
   *
   * @return names of output states.
   */
  @SuppressWarnings("unchecked")
  public Map<String, Integer> getOutputStateNames() {
    return outputStateNames;
  }

  /**
   * Register an array of hidden state Names. We assume that the state name at
   * position i has the ID i
   *
   * @param stateNames state names to register.
   */
  public void registerOutputStateNames(String[] stateNames) {
    if (stateNames != null) {
      outputStateNames = new TreeBidiMap();
      for (int i = 0; i < stateNames.length; ++i) {
        outputStateNames.put(stateNames[i], i);
      }
    }
  }

  /**
   * Register a map of hidden state Names/state IDs
   *
   * @param stateNames <String,Integer> Map that assigns each state name an integer ID
   */
  public void registerOutputStateNames(Map<String, Integer> stateNames) {
    if (stateNames != null)
      outputStateNames = new TreeBidiMap(stateNames);
  }

  /**
   * Lookup the name for the given output state id
   *
   * @param id Integer id of the output state
   * @return String containing the name for the given id, null if this id is not
   *         known or no output state names were specified
   */
  public String getOutputStateName(int id) {
    if (outputStateNames == null)
      return null;
    return (String) outputStateNames.getKey(id);
  }

  /**
   * Lookup the ID for the given output state name
   *
   * @param name Name of the output state
   * @return int containing the ID for the given name, -1 if this name is not
   *         known or no output state names were specified
   */
  public int getOutputStateID(String name) {
    if (outputStateNames == null)
      return -1;
    Integer tmp = (Integer) outputStateNames.get(name);
    return (tmp == null) ? -1 : tmp;
  }

  /**
   * Encode this HMMmodel as a JSON string
   *
   * @return String containing the JSON of this model
   */
  public String toJson() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(HmmModel.class, this);
    Gson gson = builder.create();
    return gson.toJson(this);
  }

  /**
   * Decode this HmmModel from a JSON string
   *
   * @param json String containing JSON representation of this model
   * @return Initialized model
   */
  public static HmmModel fromJson(String json) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(HmmModel.class, new HmmModel());
    Gson gson = builder.create();
    return gson.fromJson(json, HmmModel.class);
  }

  // CODE USED FOR SERIALIZATION

  private static final String MODEL = "HMMModel";
  private static final String OUTNAMES = "HMMOutNames";
  private static final String HIDDENNAMES = "HmmHiddenNames";

  /**
   * {@inheritDoc}
   */
  @Override
  public HmmModel deserialize(JsonElement json, Type type,
                              JsonDeserializationContext context) {
    // register the builders for matrix / vector
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    // now decode the original model
    JsonObject obj = json.getAsJsonObject();
    String modelString = obj.get(MODEL).getAsString();
    HmmModel model = gson.fromJson(modelString, HmmModel.class);
    // now decode the names
    JsonElement names = obj.get(HIDDENNAMES);
    if (names != null) {
      Map<String, Integer> tmpMap = gson.fromJson(names.getAsString(),
          new TypeToken<Map<String, Integer>>() {
          } .getType());
      model.registerHiddenStateNames(tmpMap);
    }
    names = obj.get(OUTNAMES);
    if (names != null) {
      Map<String, Integer> tmpMap = gson.fromJson(names.getAsString(),
          new TypeToken<Map<String, Integer>>() {
          } .getType());
      model.registerOutputStateNames(tmpMap);
    }
    // return the model
    return model;
  }

  @Override
  public JsonElement serialize(HmmModel model, Type arg1,
                               JsonSerializationContext arg2) {
    // we need to make sure that we serialize the bidimaps separately, since
    // GSON is unable to do this naively
    BidiMap hiddenNames = model.hiddenStateNames;
    model.hiddenStateNames = null;
    BidiMap outNames = model.outputStateNames;
    model.outputStateNames = null;
    // now register the builders for matrix / vector
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    // create a model
    JsonObject json = new JsonObject();
    // first, we add the model
    json.add(MODEL, new JsonPrimitive(gson.toJson(model)));
    // now, we add the state names
    if (outNames != null) {
      json.add(OUTNAMES, new JsonPrimitive(gson.toJson(outNames)));
    }
    if (hiddenNames != null) {
      json.add(HIDDENNAMES, new JsonPrimitive(gson.toJson(hiddenNames)));
    }
    // return the model to its original state :)
    model.hiddenStateNames = hiddenNames;
    model.outputStateNames = outNames;
    return json;
  }
}
