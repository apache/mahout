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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

/**
 * AbstractNeuralNetwork defines the general operations for a neural network
 * based model. Typically, all derivative models such as Multilayer Perceptron
 * and Autoencoder consist of neurons and the weights between neurons.
 */
public abstract class NeuralNetwork {

  /* The default learning rate */
  private static final double DEFAULT_LEARNING_RATE = 0.5;
  /* The default regularization weight */
  private static final double DEFAULT_REGULARIZATION_WEIGHT = 0;
  /* The default momentum weight */
  private static final double DEFAULT_MOMENTUM_WEIGHT = 0.1;

  public static enum TrainingMethod {
    GRADIENT_DESCENT
  }

  /* the name of the model */
  protected String modelType;

  /* the path to store the model */
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
  protected int finalLayerIdx;

  /**
   * The default constructor that initializes the learning rate, regularization
   * weight, and momentum weight by default.
   */
  public NeuralNetwork() {
    this.learningRate = DEFAULT_LEARNING_RATE;
    this.regularizationWeight = DEFAULT_REGULARIZATION_WEIGHT;
    this.momentumWeight = DEFAULT_MOMENTUM_WEIGHT;
    this.trainingMethod = TrainingMethod.GRADIENT_DESCENT;
    this.costFunctionName = "Minus_Squared";
    this.modelType = this.getClass().getSimpleName();

    this.layerSizeList = Lists.newArrayList();
    this.layerSizeList = Lists.newArrayList();
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
    this.squashingFunctionList = Lists.newArrayList();
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
    this.setLearningRate(learningRate);
    this.setMomentumWeight(momentumWeight);
    this.setRegularizationWeight(regularizationWeight);
  }

  /**
   * Initialize the NeuralNetwork by specifying the location of the model.
   * 
   * @param modelPath The location that the model is stored.
   */
  public NeuralNetwork(String modelPath) {
    try {
      this.modelPath = modelPath;
      this.readFromModel();
    } catch (IOException e) {
      e.printStackTrace();
    }
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
  public NeuralNetwork setLearningRate(double learningRate) {
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
    return this.learningRate;
  }

  /**
   * Set the regularization weight. More complex the model is, less weight the
   * regularization is.
   * 
   * @param regularizationWeight regularization must be in the range [0, 0.1).
   * @return The model instance.
   */
  public NeuralNetwork setRegularizationWeight(double regularizationWeight) {
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
    return this.regularizationWeight;
  }

  /**
   * Set the momentum weight for the model.
   * 
   * @param momentumWeight momentumWeight must be in range [0, 0.5].
   * @return The model instance.
   */
  public NeuralNetwork setMomentumWeight(double momentumWeight) {
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
    return this.momentumWeight;
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
    return this.trainingMethod;
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
    int actualSize = size;
    if (!isFinalLayer) {
      actualSize += 1;
    }

    this.layerSizeList.add(actualSize);
    int layerIdx = this.layerSizeList.size() - 1;
    if (isFinalLayer) {
      this.finalLayerIdx = layerIdx;
    }

    // add weights between current layer and previous layer, and input layer has
    // no squashing function
    if (layerIdx > 0) {
      int sizePrevLayer = this.layerSizeList.get(layerIdx - 1);
      // row count equals to size of current size and column count equal to
      // size of previous layer
      int row = isFinalLayer ? actualSize : actualSize - 1;
      Matrix weightMatrix = new DenseMatrix(row, sizePrevLayer);
      // initialize weights
      final RandomWrapper rnd = RandomUtils.getRandom();
      weightMatrix.assign(new DoubleFunction() {
        @Override
        public double apply(double value) {
          return rnd.nextDouble() - 0.5;
        }
      });
      this.weightMatrixList.add(weightMatrix);
      this.prevWeightUpdatesList.add(new DenseMatrix(row, sizePrevLayer));
      this.squashingFunctionList.add(squashingFunctionName);
    }
    return layerIdx;
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
    return this.layerSizeList.get(layer);
  }

  /**
   * Get the layer size list.
   * 
   * @return The sizes of the layers.
   */
  protected List<Integer> getLayerSizeList() {
    return this.layerSizeList;
  }

  /**
   * Get the weights between layer layerIdx and layerIdx + 1
   * 
   * @param layerIdx The index of the layer.
   * @return The weights in form of {@link Matrix}.
   */
  public Matrix getWeightsByLayer(int layerIdx) {
    return this.weightMatrixList.get(layerIdx);
  }

  /**
   * Update the weight matrices with given matrices.
   * 
   * @param matrices The weight matrices, must be the same dimension as the
   *          existing matrices.
   */
  public void updateWeightMatrices(Matrix[] matrices) {
    for (int i = 0; i < matrices.length; ++i) {
      Matrix matrix = this.weightMatrixList.get(i);
      this.weightMatrixList.set(i, matrix.plus(matrices[i]));
    }
  }

  /**
   * Set the weight matrices.
   * 
   * @param matrices The weight matrices, must be the same dimension of the
   *          existing matrices.
   */
  public void setWeightMatrices(Matrix[] matrices) {
    this.weightMatrixList = Lists.newArrayList();
    Collections.addAll(this.weightMatrixList, matrices);
  }

  /**
   * Set the weight matrix for a specified layer.
   * 
   * @param index The index of the matrix, starting from 0 (between layer 0 and 1).
   * @param matrix The instance of {@link Matrix}.
   */
  public void setWeightMatrix(int index, Matrix matrix) {
    Preconditions.checkArgument(0 <= index && index < this.weightMatrixList.size(),
        String.format("index [%s] should be in range [%s, %s).", index, 0, this.weightMatrixList.size()));
    this.weightMatrixList.set(index, matrix);
  }

  /**
   * Get all the weight matrices.
   * 
   * @return The weight matrices.
   */
  public Matrix[] getWeightMatrices() {
    Matrix[] matrices = new Matrix[this.weightMatrixList.size()];
    this.weightMatrixList.toArray(matrices);
    return matrices;
  }

  /**
   * Get the output calculated by the model.
   * 
   * @param instance The feature instance in form of {@link Vector}, each dimension contains the value of the corresponding feature.
   * @return The output vector.
   */
  public Vector getOutput(Vector instance) {
    Preconditions.checkArgument(this.layerSizeList.get(0) == instance.size() + 1,
        String.format("The dimension of input instance should be %d, but the input has dimension %d.",
            this.layerSizeList.get(0) - 1, instance.size()));

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

    for (int i = 0; i < this.layerSizeList.size() - 1; ++i) {
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
    Matrix weightMatrix = this.weightMatrixList.get(fromLayer);

    Vector vec = weightMatrix.times(intermediateOutput);
    vec = vec.assign(NeuralNetworkFunctions.getDoubleFunction(this.squashingFunctionList.get(fromLayer)));

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
    Matrix[] matrices = this.trainByInstance(trainingInstance);
    this.updateWeightMatrices(matrices);
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
    int inputDimension = this.layerSizeList.get(0) - 1;
    int outputDimension = this.layerSizeList.get(this.layerSizeList.size() - 1);
    Preconditions.checkArgument(inputDimension + outputDimension == trainingInstance.size(),
        String.format("The dimension of training instance is %d, but requires %d.", trainingInstance.size(),
            inputDimension + outputDimension));

    if (this.trainingMethod.equals(TrainingMethod.GRADIENT_DESCENT)) {
      return this.trainByInstanceGradientDescent(trainingInstance);
    }
    throw new IllegalArgumentException(String.format("Training method is not supported."));
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
    int inputDimension = this.layerSizeList.get(0) - 1;

    Vector inputInstance = new DenseVector(this.layerSizeList.get(0));
    inputInstance.set(0, 1); // add bias
    for (int i = 0; i < inputDimension; ++i) {
      inputInstance.set(i + 1, trainingInstance.get(i));
    }

    Vector labels = trainingInstance.viewPart(inputInstance.size() - 1, trainingInstance.size() - inputInstance.size() + 1);

    // initialize weight update matrices
    Matrix[] weightUpdateMatrices = new Matrix[this.weightMatrixList.size()];
    for (int m = 0; m < weightUpdateMatrices.length; ++m) {
      weightUpdateMatrices[m] = new DenseMatrix(this.weightMatrixList.get(m).rowSize(), this.weightMatrixList.get(m).columnSize());
    }

    List<Vector> internalResults = this.getOutputInternal(inputInstance);

    Vector deltaVec = new DenseVector(this.layerSizeList.get(this.layerSizeList.size() - 1));
    Vector output = internalResults.get(internalResults.size() - 1);

    final DoubleFunction derivativeSquashingFunction =
        NeuralNetworkFunctions.getDerivativeDoubleFunction(this.squashingFunctionList.get(this.squashingFunctionList.size() - 1));

    final DoubleDoubleFunction costFunction = NeuralNetworkFunctions.getDerivativeDoubleDoubleFunction(this.costFunctionName);

    Matrix lastWeightMatrix = this.weightMatrixList.get(this.weightMatrixList.size() - 1);

    for (int i = 0; i < deltaVec.size(); ++i) {
      double costFuncDerivative = costFunction.apply(labels.get(i), output.get(i + 1));
      // add regularization
      costFuncDerivative += this.regularizationWeight * lastWeightMatrix.viewRow(i).zSum();
      deltaVec.set(i, costFuncDerivative);
      deltaVec.set(i, deltaVec.get(i) * derivativeSquashingFunction.apply(output.get(i + 1)));
    }

    // start from previous layer of output layer
    for (int layer = this.layerSizeList.size() - 2; layer >= 0; --layer) {
      deltaVec = backPropagate(layer, deltaVec, internalResults, weightUpdateMatrices[layer]);
    }

    this.prevWeightUpdatesList = Arrays.asList(weightUpdateMatrices);

    return weightUpdateMatrices;
  }

  /**
   * Back-propagate the errors to from next layer to current layer. The weight
   * updated information will be stored in the weightUpdateMatrices, and the
   * delta of the prevLayer will be returned.
   * 
   * @param curLayerIdx Index of current layer.
   * @param nextLayerDelta Delta of next layer.
   * @param outputCache The output cache to store intermediate results.
   * @param weightUpdateMatrix  The weight update, in form of {@link Matrix}.
   * @return The weight updates.
   */
  private Vector backPropagate(int curLayerIdx, Vector nextLayerDelta,
                               List<Vector> outputCache, Matrix weightUpdateMatrix) {

    // get layer related information
    final DoubleFunction derivativeSquashingFunction =
        NeuralNetworkFunctions.getDerivativeDoubleFunction(this.squashingFunctionList.get(curLayerIdx));
    Vector curLayerOutput = outputCache.get(curLayerIdx);
    Matrix weightMatrix = this.weightMatrixList.get(curLayerIdx);
    Matrix prevWeightMatrix = this.prevWeightUpdatesList.get(curLayerIdx);

    // next layer is not output layer, remove the delta of bias neuron
    if (curLayerIdx != this.layerSizeList.size() - 2) {
      nextLayerDelta = nextLayerDelta.viewPart(1, nextLayerDelta.size() - 1);
    }

    Vector delta = weightMatrix.transpose().times(nextLayerDelta);

    delta = delta.assign(curLayerOutput, new DoubleDoubleFunction() {
      @Override
      public double apply(double deltaElem, double curLayerOutputElem) {
        return deltaElem * derivativeSquashingFunction.apply(curLayerOutputElem);
      }
    });

    // update weights
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
    Preconditions.checkArgument(this.modelPath != null, "Model path has not been set.");
    FSDataInputStream is = null;
    try {
      Path path = new Path(this.modelPath);
      FileSystem fs = path.getFileSystem(new Configuration());
      is = new FSDataInputStream(fs.open(path));
      this.readFields(is);
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
    Preconditions.checkArgument(this.modelPath != null, "Model path has not been set.");
    FSDataOutputStream stream = null;
    try {
      Path path = new Path(this.modelPath);
      FileSystem fs = path.getFileSystem(new Configuration());
      stream = fs.create(path, true);
      this.write(stream);
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
    return this.modelPath;
  }

  /**
   * Write the fields of the model to output.
   * 
   * @param output The output instance.
   * @throws IOException
   */
  public void write(DataOutput output) throws IOException {
    // write model type
    WritableUtils.writeString(output, modelType);
    // write learning rate
    output.writeDouble(learningRate);
    // write model path
    if (this.modelPath != null) {
      WritableUtils.writeString(output, modelPath);
    } else {
      WritableUtils.writeString(output, "null");
    }

    // write regularization weight
    output.writeDouble(this.regularizationWeight);
    // write momentum weight
    output.writeDouble(this.momentumWeight);

    // write cost function
    WritableUtils.writeString(output, this.costFunctionName);

    // write layer size list
    output.writeInt(this.layerSizeList.size());
    for (Integer aLayerSizeList : this.layerSizeList) {
      output.writeInt(aLayerSizeList);
    }

    WritableUtils.writeEnum(output, this.trainingMethod);

    // write squashing functions
    output.writeInt(this.squashingFunctionList.size());
    for (String aSquashingFunctionList : this.squashingFunctionList) {
      WritableUtils.writeString(output, aSquashingFunctionList);
    }

    // write weight matrices
    output.writeInt(this.weightMatrixList.size());
    for (Matrix aWeightMatrixList : this.weightMatrixList) {
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
    // read model type
    this.modelType = WritableUtils.readString(input);
    if (!this.modelType.equals(this.getClass().getSimpleName())) {
      throw new IllegalArgumentException("The specified location does not contains the valid NeuralNetwork model.");
    }
    // read learning rate
    this.learningRate = input.readDouble();
    // read model path
    this.modelPath = WritableUtils.readString(input);
    if (this.modelPath.equals("null")) {
      this.modelPath = null;
    }

    // read regularization weight
    this.regularizationWeight = input.readDouble();
    // read momentum weight
    this.momentumWeight = input.readDouble();

    // read cost function
    this.costFunctionName = WritableUtils.readString(input);

    // read layer size list
    int numLayers = input.readInt();
    this.layerSizeList = Lists.newArrayList();
    for (int i = 0; i < numLayers; i++) {
      this.layerSizeList.add(input.readInt());
    }

    this.trainingMethod = WritableUtils.readEnum(input, TrainingMethod.class);

    // read squash functions
    int squashingFunctionSize = input.readInt();
    this.squashingFunctionList = Lists.newArrayList();
    for (int i = 0; i < squashingFunctionSize; i++) {
      this.squashingFunctionList.add(WritableUtils.readString(input));
    }

    // read weights and construct matrices of previous updates
    int numOfMatrices = input.readInt();
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
    for (int i = 0; i < numOfMatrices; i++) {
      Matrix matrix = MatrixWritable.readMatrix(input);
      this.weightMatrixList.add(matrix);
      this.prevWeightUpdatesList.add(new DenseMatrix(matrix.rowSize(), matrix.columnSize()));
    }
  }

}