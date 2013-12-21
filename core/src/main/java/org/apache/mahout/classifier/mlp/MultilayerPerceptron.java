/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * A Multilayer Perceptron (MLP) is a kind of feed-forward artificial neural
 * network, which is a mathematical model inspired by the biological neural
 * network. The Multilayer Perceptron can be used for various machine learning
 * tasks such as classification and regression.
 * 
 * A detailed introduction about MLP can be found at
 * http://ufldl.stanford.edu/wiki/index.php/Neural_Networks.
 * 
 * For this particular implementation, the users can freely control the topology
 * of the MLP, including: 1. The size of the input layer; 2. The number of
 * hidden layers; 3. The size of each hidden layer; 4. The size of the output
 * layer. 5. The cost function. 6. The squashing function.
 * 
 * The model is trained in an online learning approach, where the weights of
 * neurons in the MLP is updated incremented using backPropagation algorithm
 * proposed by (Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986)
 * Learning representations by back-propagating errors. Nature, 323, 533--536.)
 */
public class MultilayerPerceptron extends NeuralNetwork implements OnlineLearner {

  /**
   * The default constructor.
   */
  public MultilayerPerceptron() {
    super();
  }

  /**
   * Initialize the MLP by specifying the location of the model.
   * 
   * @param modelPath The path of the model.
   */
  public MultilayerPerceptron(String modelPath) {
    super(modelPath);
  }

  @Override
  public void train(int actual, Vector instance) {
    // construct the training instance, where append the actual to instance
    Vector trainingInstance = new DenseVector(instance.size() + 1);
    for (int i = 0; i < instance.size(); ++i) {
      trainingInstance.setQuick(i, instance.getQuick(i));
    }
    trainingInstance.setQuick(instance.size(), actual);
    this.trainOnline(trainingInstance);
  }

  @Override
  public void train(long trackingKey, String groupKey, int actual,
      Vector instance) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void close() {
    // DO NOTHING
  }

}
