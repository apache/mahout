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
package org.apache.mahout.classifier.mlp;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.WritableUtils;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

/**
 * AbstractNeuralNetwork defines the general operations for a neural network
 * based model. Typically, all derivative models such as Multilayer Perceptron
 * and Autoencoder consist of neurons and the weights between neurons.
 */
public abstract class NeuralNetwork {
  
  private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

  /* The default learning rate */
  public static final double DEFAULT_LEARNING_RATE = 0.5;
  /* The default regularization weight */
  public static final double DEFAULT_REGULARIZATION_WEIGHT = 0;
  /* The default momentum weight */
  public static final double DEFAULT_MOMENTUM_WEIGHT = 0.1;

  public static enum TrainingMethod { GRADIENT_DESCENT }

  /* The name of the model */
  protected String modelType;

  /* The path to store the model */
  protected String modelPath;

  protected double learningRate;

  /* The weight of regularization */
  protected double regularizationWeight;

  /* The momentum weight */
  protected double momentumWeight;

  /* The cost function of the model */
  protected String costFunctionName;

  /* Record the size of each layer */
  protected List<Integer> layerSizeList;

  /* Training method used for training the model */
  protected TrainingMethod trainingMethod;

  /* Weights between neurons at adjacent layers */
  protected List<Matrix> weightMatrixList;

  /* Previous weight updates between neurons at adjacent layers */
  protected List<Matrix> prevWeightUpdatesList;

  /* Different layers can have different squashing function */
  protected List<String> squashingFunctionList;

  /* The index of the final layer */
  protected int finalLayerIndex;

  /**
   * The default constructor that initializes the learning rate, regularization
   * weight, and momentum weight by default.
   */
  public NeuralNetwork() {
    log.info("Initialize model...");
    learningRate = DEFAULT_LEARNING_RATE;
    regularizationWeight = DEFAULT_REGULARIZATION_WEIGHT;
    momentumWeight = DEFAULT_MOMENTUM_WEIGHT;
    trainingMethod = TrainingMethod.GRADIENT_DESCENT;
    costFunctionName = "Minus_Squared";
    modelType = getClass().getSimpleName();

    layerSizeList = Lists.newArrayList();
    layerSizeList = Lists.newArrayList();
    weightMatrixList = Lists.newArrayList();
    prevWeightUpdatesList = Lists.newArrayList();
    squashingFunctionList = Lists.newArrayList();
  }

  /**
   * Initialize the NeuralNetwork by specifying learning rate, momentum weight
   * and regularization weight.
   * 
   * @param learningRate The learning rate.
   * @param momentumWeight The momentum weight.
   * @param regularizationWeight The regularization weight.
   */
  public NeuralNetwork(double learningRate, double momentumWeight, double regularizationWeight) {
    this();
    setLearningRate(learningRate);
    setMomentumWeight(momentumWeight);
    setRegularizationWeight(regularizationWeight);
  }

  /**
   * Initialize the NeuralNetwork by specifying the location of the model.
   * 
   * @param modelPath The location that the model is stored.
   */
  public NeuralNetwork(String modelPath) throws IOException {
    this.modelPath = modelPath;
    readFromModel();
  }

  /**
   * Get the type of the model.
   * 
   * @return The name of the model.
   */
  public String getModelType() {
    return this.modelType;
  }

  /**
   * Set the degree of aggression during model training, a large learning rate
   * can increase the training speed, but it also decreases the chance of model
   * converge.
   * 
   * @param learningRate Learning rate must be a non-negative value. Recommend in range (0, 0.5).
   * @return The model instance.
   */
  public final NeuralNetwork setLearningRate(double learningRate) {
    Preconditions.checkArgument(learningRate > 0, "Learning rate must be larger than 0.");
    this.learningRate = learningRate;
    return this;
  }

  /**
   * Get the value of learning rate.
   * 
   * @return The value of learning rate.
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * Set the regularization weight. More complex the model is, less weight the
   * regularization is.
   * 
   * @param regularizationWeight regularization must be in the range [0, 0.1).
   * @return The model instance.
   */
  public final NeuralNetwork setRegularizationWeight(double regularizationWeight) {
    Preconditions.checkArgument(regularizationWeight >= 0
        && regularizationWeight < 0.1, "Regularization weight must be in range [0, 0.1)");
    this.regularizationWeight = regularizationWeight;
    return this;
  }

  /**
   * Get the weight of the regularization.
   * 
   * @return The weight of regularization.
   */
  public double getRegularizationWeight() {
    return regularizationWeight;
  }

  /**
   * Set the momentum weight for the model.
   * 
   * @param momentumWeight momentumWeight must be in range [0, 0.5].
   * @return The model instance.
   */
  public final NeuralNetwork setMomentumWeight(double momentumWeight) {
    Preconditions.checkArgument(momentumWeight >= 0 && momentumWeight <= 1.0,
        "Momentum weight must be in range [0, 1.0]");
    this.momentumWeight = momentumWeight;
    return this;
  }

  /**
   * Get the momentum weight.
   * 
   * @return The value of momentum.
   */
  public double getMomentumWeight() {
    return momentumWeight;
  }

  /**
   * Set the training method.
   * 
   * @param method The training method, currently supports GRADIENT_DESCENT.
   * @return The instance of the model.
   */
  public NeuralNetwork setTrainingMethod(TrainingMethod method) {
    this.trainingMethod = method;
    return this;
  }

  /**
   * Get the training method.
   * 
   * @return The training method enumeration.
   */
  public TrainingMethod getTrainingMethod() {
    return trainingMethod;
  }

  /**
   * Set the cost function for the model.
   * 
   * @param costFunction the name of the cost function. Currently supports
   *          "Minus_Squared", "Cross_Entropy".
   */
  public NeuralNetwork setCostFunction(String costFunction) {
    this.costFunctionName = costFunction;
    return this;
  }

  /**
   * Add a layer of neurons with specified size. If the added layer is not the
   * first layer, it will automatically connect the neurons between with the
   * previous layer.
   * 
   * @param size The size of the layer. (bias neuron excluded)
   * @param isFinalLayer If false, add a bias neuron.
   * @param squashingFunctionName The squashing function for this layer, input
   *          layer is f(x) = x by default.
   * @return The layer index, starts with 0.
   */
  public int addLayer(int size, boolean isFinalLayer, String squashingFunctionName) {
    Preconditions.checkArgument(size > 0, "Size of layer must be larger than 0.");
    log.info("Add layer with size {} and squashing function {}", size, squashingFunctionName);
    int actualSize = size;
    if (!isFinalLayer) {
      actualSize += 1;
    }

    layerSizeList.add(actualSize);
    int layerIndex = layerSizeList.size() - 1;
    if (isFinalLayer) {
      finalLayerIndex = layerIndex;
    }

    // Add weights between current layer and previous layer, and input layer has no squashing function
    if (layerIndex > 0) {
      int sizePrevLayer = layerSizeList.get(layerIndex - 1);
      // Row count equals to size of current size and column count equal to size of previous layer
      int row = isFinalLayer ? actualSize : actualSize - 1;
      Matrix weightMatrix = new DenseMatrix(row, sizePrevLayer);
      // Initialize weights
      final RandomWrapper rnd = RandomUtils.getRandom();
      weightMatrix.assign(new DoubleFunction() {
        @Override
        public double apply(double value) {
          return rnd.nextDouble() - 0.5;
        }
      });
      weightMatrixList.add(weightMatrix);
      prevWeightUpdatesList.add(new DenseMatrix(row, sizePrevLayer));
      squashingFunctionList.add(squashingFunctionName);
    }
    return layerIndex;
  }

  /**
   * Get the size of a particular layer.
   * 
   * @param layer The index of the layer, starting from 0.
   * @return The size of the corresponding layer.
   */
  public int getLayerSize(int layer) {
    Preconditions.checkArgument(layer >= 0 && layer < this.layerSizeList.size(),
        String.format("Input must be in range [0, %d]\n", this.layerSizeList.size() - 1));
    return layerSizeList.get(layer);
  }

  /**
   * Get the layer size list.
   * 
   * @return The sizes of the layers.
   */
  protected List<Integer> getLayerSizeList() {
    return layerSizeList;
  }

  /**
   * Get the weights between layer layerIndex and layerIndex + 1
   * 
   * @param layerIndex The index of the layer.
   * @return The weights in form of {@link Matrix}.
   */
  public Matrix getWeightsByLayer(int layerIndex) {
    return weightMatrixList.get(layerIndex);
  }

  /**
   * Update the weight matrices with given matrices.
   * 
   * @param matrices The weight matrices, must be the same dimension as the
   *          existing matrices.
   */
  public void updateWeightMatrices(Matrix[] matrices) {
    for (int i = 0; i < matrices.length; ++i) {
      Matrix matrix = weightMatrixList.get(i);
      weightMatrixList.set(i, matrix.plus(matrices[i]));
    }
  }

  /**
   * Set the weight matrices.
   * 
   * @param matrices The weight matrices, must be the same dimension of the
   *          existing matrices.
   */
  public void setWeightMatrices(Matrix[] matrices) {
    weightMatrixList = Lists.newArrayList();
    Collections.addAll(weightMatrixList, matrices);
  }

  /**
   * Set the weight matrix for a specified layer.
   * 
   * @param index The index of the matrix, starting from 0 (between layer 0 and 1).
   * @param matrix The instance of {@link Matrix}.
   */
  public void setWeightMatrix(int index, Matrix matrix) {
    Preconditions.checkArgument(0 <= index && index < weightMatrixList.size(),
        String.format("index [%s] should be in range [%s, %s).", index, 0, weightMatrixList.size()));
    weightMatrixList.set(index, matrix);
  }

  /**
   * Get all the weight matrices.
   * 
   * @return The weight matrices.
   */
  public Matrix[] getWeightMatrices() {
    Matrix[] matrices = new Matrix[weightMatrixList.size()];
    weightMatrixList.toArray(matrices);
    return matrices;
  }

  /**
   * Get the output calculated by the model.
   * 
   * @param instance The feature instance in form of {@link Vector}, each dimension contains the value of the corresponding feature.
   * @return The output vector.
   */
  public Vector getOutput(Vector instance) {
    Preconditions.checkArgument(layerSizeList.get(0) == instance.size() + 1,
        String.format("The dimension of input instance should be %d, but the input has dimension %d.",
            layerSizeList.get(0) - 1, instance.size()));

    // add bias feature
    Vector instanceWithBias = new DenseVector(instance.size() + 1);
    // set bias to be a little bit less than 1.0
    instanceWithBias.set(0, 0.99999);
    for (int i = 1; i < instanceWithBias.size(); ++i) {
      instanceWithBias.set(i, instance.get(i - 1));
    }

    List<Vector> outputCache = getOutputInternal(instanceWithBias);
    // return the output of the last layer
    Vector result = outputCache.get(outputCache.size() - 1);
    // remove bias
    return result.viewPart(1, result.size() - 1);
  }

  /**
   * Calculate output internally, the intermediate output of each layer will be
   * stored.
   * 
   * @param instance The feature instance in form of {@link Vector}, each dimension contains the value of the corresponding feature.
   * @return Cached output of each layer.
   */
  protected List<Vector> getOutputInternal(Vector instance) {
    List<Vector> outputCache = Lists.newArrayList();
    // fill with instance
    Vector intermediateOutput = instance;
    outputCache.add(intermediateOutput);

    for (int i = 0; i < layerSizeList.size() - 1; ++i) {
      intermediateOutput = forward(i, intermediateOutput);
      outputCache.add(intermediateOutput);
    }
    return outputCache;
  }

  /**
   * Forward the calculation for one layer.
   * 
   * @param fromLayer The index of the previous layer.
   * @param intermediateOutput The intermediate output of previous layer.
   * @return The intermediate results of the current layer.
   */
  protected Vector forward(int fromLayer, Vector intermediateOutput) {
    Matrix weightMatrix = weightMatrixList.get(fromLayer);

    Vector vec = weightMatrix.times(intermediateOutput);
    vec = vec.assign(NeuralNetworkFunctions.getDoubleFunction(squashingFunctionList.get(fromLayer)));

    // add bias
    Vector vecWithBias = new DenseVector(vec.size() + 1);
    vecWithBias.set(0, 1);
    for (int i = 0; i < vec.size(); ++i) {
      vecWithBias.set(i + 1, vec.get(i));
    }
    return vecWithBias;
  }

  /**
   * Train the neural network incrementally with given training instance.
   * 
   * @param trainingInstance An training instance, including the features and the label(s). Its dimension must equals
   *          to the size of the input layer (bias neuron excluded) + the size
   *          of the output layer (a.k.a. the dimension of the labels).
   */
  public void trainOnline(Vector trainingInstance) {
    Matrix[] matrices = trainByInstance(trainingInstance);
    updateWeightMatrices(matrices);
  }

  /**
   * Get the updated weights using one training instance.
   * 
   * @param trainingInstance An training instance, including the features and the label(s). Its dimension must equals
   *          to the size of the input layer (bias neuron excluded) + the size
   *          of the output layer (a.k.a. the dimension of the labels).
   * @return The update of each weight, in form of {@link Matrix} list.
   */
  public Matrix[] trainByInstance(Vector trainingInstance) {
    // validate training instance
    int inputDimension = layerSizeList.get(0) - 1;
    int outputDimension = layerSizeList.get(this.layerSizeList.size() - 1);
    Preconditions.checkArgument(inputDimension + outputDimension == trainingInstance.size(),
        String.format("The dimension of training instance is %d, but requires %d.", trainingInstance.size(),
            inputDimension + outputDimension));

    if (trainingMethod.equals(TrainingMethod.GRADIENT_DESCENT)) {
      return trainByInstanceGradientDescent(trainingInstance);
    }
    throw new IllegalArgumentException("Training method is not supported.");
  }

  /**
   * Train by gradient descent. Get the updated weights using one training
   * instance.
   * 
   * @param trainingInstance An training instance, including the features and the label(s). Its dimension must equals
   *          to the size of the input layer (bias neuron excluded) + the size
   *          of the output layer (a.k.a. the dimension of the labels).
   * @return The weight update matrices.
   */
  private Matrix[] trainByInstanceGradientDescent(Vector trainingInstance) {
    int inputDimension = layerSizeList.get(0) - 1;

    Vector inputInstance = new DenseVector(layerSizeList.get(0));
    inputInstance.set(0, 1); // add bias
    for (int i = 0; i < inputDimension; ++i) {
      inputInstance.set(i + 1, trainingInstance.get(i));
    }

    Vector labels =
        trainingInstance.viewPart(inputInstance.size() - 1, trainingInstance.size() - inputInstance.size() + 1);

    // initialize weight update matrices
    Matrix[] weightUpdateMatrices = new Matrix[weightMatrixList.size()];
    for (int m = 0; m < weightUpdateMatrices.length; ++m) {
      weightUpdateMatrices[m] =
          new DenseMatrix(weightMatrixList.get(m).rowSize(), weightMatrixList.get(m).columnSize());
    }

    List<Vector> internalResults = getOutputInternal(inputInstance);

    Vector deltaVec = new DenseVector(layerSizeList.get(layerSizeList.size() - 1));
    Vector output = internalResults.get(internalResults.size() - 1);

    final DoubleFunction derivativeSquashingFunction =
        NeuralNetworkFunctions.getDerivativeDoubleFunction(squashingFunctionList.get(squashingFunctionList.size() - 1));

    final DoubleDoubleFunction costFunction =
        NeuralNetworkFunctions.getDerivativeDoubleDoubleFunction(costFunctionName);

    Matrix lastWeightMatrix = weightMatrixList.get(weightMatrixList.size() - 1);

    for (int i = 0; i < deltaVec.size(); ++i) {
      double costFuncDerivative = costFunction.apply(labels.get(i), output.get(i + 1));
      // Add regularization
      costFuncDerivative += regularizationWeight * lastWeightMatrix.viewRow(i).zSum();
      deltaVec.set(i, costFuncDerivative);
      deltaVec.set(i, deltaVec.get(i) * derivativeSquashingFunction.apply(output.get(i + 1)));
    }

    // Start from previous layer of output layer
    for (int layer = layerSizeList.size() - 2; layer >= 0; --layer) {
      deltaVec = backPropagate(layer, deltaVec, internalResults, weightUpdateMatrices[layer]);
    }

    prevWeightUpdatesList = Arrays.asList(weightUpdateMatrices);

    return weightUpdateMatrices;
  }

  /**
   * Back-propagate the errors to from next layer to current layer. The weight
   * updated information will be stored in the weightUpdateMatrices, and the
   * delta of the prevLayer will be returned.
   * 
   * @param currentLayerIndex Index of current layer.
   * @param nextLayerDelta Delta of next layer.
   * @param outputCache The output cache to store intermediate results.
   * @param weightUpdateMatrix  The weight update, in form of {@link Matrix}.
   * @return The weight updates.
   */
  private Vector backPropagate(int currentLayerIndex, Vector nextLayerDelta,
                               List<Vector> outputCache, Matrix weightUpdateMatrix) {

    // Get layer related information
    final DoubleFunction derivativeSquashingFunction =
        NeuralNetworkFunctions.getDerivativeDoubleFunction(squashingFunctionList.get(currentLayerIndex));
    Vector curLayerOutput = outputCache.get(currentLayerIndex);
    Matrix weightMatrix = weightMatrixList.get(currentLayerIndex);
    Matrix prevWeightMatrix = prevWeightUpdatesList.get(currentLayerIndex);

    // Next layer is not output layer, remove the delta of bias neuron
    if (currentLayerIndex != layerSizeList.size() - 2) {
      nextLayerDelta = nextLayerDelta.viewPart(1, nextLayerDelta.size() - 1);
    }

    Vector delta = weightMatrix.transpose().times(nextLayerDelta);

    delta = delta.assign(curLayerOutput, new DoubleDoubleFunction() {
      @Override
      public double apply(double deltaElem, double curLayerOutputElem) {
        return deltaElem * derivativeSquashingFunction.apply(curLayerOutputElem);
      }
    });

    // Update weights
    for (int i = 0; i < weightUpdateMatrix.rowSize(); ++i) {
      for (int j = 0; j < weightUpdateMatrix.columnSize(); ++j) {
        weightUpdateMatrix.set(i, j, -learningRate * nextLayerDelta.get(i) *
            curLayerOutput.get(j) + this.momentumWeight * prevWeightMatrix.get(i, j));
      }
    }

    return delta;
  }

  /**
   * Read the model meta-data from the specified location.
   * 
   * @throws IOException
   */
  protected void readFromModel() throws IOException {
    log.info("Load model from {}", modelPath);
    Preconditions.checkArgument(modelPath != null, "Model path has not been set.");
    FSDataInputStream is = null;
    try {
      Path path = new Path(modelPath);
      FileSystem fs = path.getFileSystem(new Configuration());
      is = new FSDataInputStream(fs.open(path));
      readFields(is);
    } finally {
      Closeables.close(is, true);
    }
  }

  /**
   * Write the model data to specified location.
   * 
   * @throws IOException
   */
  public void writeModelToFile() throws IOException {
    log.info("Write model to {}.", modelPath);
    Preconditions.checkArgument(modelPath != null, "Model path has not been set.");
    FSDataOutputStream stream = null;
    try {
      Path path = new Path(modelPath);
      FileSystem fs = path.getFileSystem(new Configuration());
      stream = fs.create(path, true);
      write(stream);
    } finally {
      Closeables.close(stream, false);
    }
  }

  /**
   * Set the model path.
   * 
   * @param modelPath The path of the model.
   */
  public void setModelPath(String modelPath) {
    this.modelPath = modelPath;
  }

  /**
   * Get the model path.
   * 
   * @return The path of the model.
   */
  public String getModelPath() {
    return modelPath;
  }

  /**
   * Write the fields of the model to output.
   * 
   * @param output The output instance.
   * @throws IOException
   */
  public void write(DataOutput output) throws IOException {
    // Write model type
    WritableUtils.writeString(output, modelType);
    // Write learning rate
    output.writeDouble(learningRate);
    // Write model path
    if (modelPath != null) {
      WritableUtils.writeString(output, modelPath);
    } else {
      WritableUtils.writeString(output, "null");
    }

    // Write regularization weight
    output.writeDouble(regularizationWeight);
    // Write momentum weight
    output.writeDouble(momentumWeight);

    // Write cost function
    WritableUtils.writeString(output, costFunctionName);

    // Write layer size list
    output.writeInt(layerSizeList.size());
    for (Integer aLayerSizeList : layerSizeList) {
      output.writeInt(aLayerSizeList);
    }

    WritableUtils.writeEnum(output, trainingMethod);

    // Write squashing functions
    output.writeInt(squashingFunctionList.size());
    for (String aSquashingFunctionList : squashingFunctionList) {
      WritableUtils.writeString(output, aSquashingFunctionList);
    }

    // Write weight matrices
    output.writeInt(this.weightMatrixList.size());
    for (Matrix aWeightMatrixList : weightMatrixList) {
      MatrixWritable.writeMatrix(output, aWeightMatrixList);
    }
  }

  /**
   * Read the fields of the model from input.
   * 
   * @param input The input instance.
   * @throws IOException
   */
  public void readFields(DataInput input) throws IOException {
    // Read model type
    modelType = WritableUtils.readString(input);
    if (!modelType.equals(this.getClass().getSimpleName())) {
      throw new IllegalArgumentException("The specified location does not contains the valid NeuralNetwork model.");
    }
    // Read learning rate
    learningRate = input.readDouble();
    // Read model path
    modelPath = WritableUtils.readString(input);
    if (modelPath.equals("null")) {
      modelPath = null;
    }

    // Read regularization weight
    regularizationWeight = input.readDouble();
    // Read momentum weight
    momentumWeight = input.readDouble();

    // Read cost function
    costFunctionName = WritableUtils.readString(input);

    // Read layer size list
    int numLayers = input.readInt();
    layerSizeList = Lists.newArrayList();
    for (int i = 0; i < numLayers; i++) {
      layerSizeList.add(input.readInt());
    }

    trainingMethod = WritableUtils.readEnum(input, TrainingMethod.class);

    // Read squash functions
    int squashingFunctionSize = input.readInt();
    squashingFunctionList = Lists.newArrayList();
    for (int i = 0; i < squashingFunctionSize; i++) {
      squashingFunctionList.add(WritableUtils.readString(input));
    }

    // Read weights and construct matrices of previous updates
    int numOfMatrices = input.readInt();
    weightMatrixList = Lists.newArrayList();
    prevWeightUpdatesList = Lists.newArrayList();
    for (int i = 0; i < numOfMatrices; i++) {
      Matrix matrix = MatrixWritable.readMatrix(input);
      weightMatrixList.add(matrix);
      prevWeightUpdatesList.add(new DenseMatrix(matrix.rowSize(), matrix.columnSize()));
    }
  }

}