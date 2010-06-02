/* Licensed to the Apache Software Foundation (ASF) under one
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
package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classifies a data point using a hyperplane.
 */
public class LinearModel {

  private static final Logger log = LoggerFactory.getLogger(LinearModel.class);

  /** Represents the direction of the hyperplane found during training.*/
  private Vector hyperplane;
  /** Displacement of hyperplane from origin.*/
  private double bias;
  /** Classification threshold. */
  private final double threshold;
  
  /**
   * Init a linear model with a hyperplane, distance and displacement.
   */
  public LinearModel(Vector hyperplane, double displacement, double threshold) {
    this.hyperplane = hyperplane;
    this.bias = displacement;
    this.threshold = threshold;
  }
  
  /**
   * Init a linear model with zero displacement and a threshold of 0.5.
   */
  public LinearModel(Vector hyperplane) {
    this(hyperplane, 0, 0.5);
  }
  
  /**
   * Classify a point to either belong to the class modeled by this linear model or not.
   * @param dataPoint the data point to classify.
   * @return returns true if data point should be classified as belonging to this model.
   */
  public boolean classify(Vector dataPoint) {
    double product = this.hyperplane.dot(dataPoint);
    if (log.isDebugEnabled()) {
      log.debug("model: {} product: {} Bias: {} threshold: {}",
                new Object[] {this, product, bias, threshold});
    }
    return product + this.bias > this.threshold;
  }
  
  /**
   * Update the hyperplane by adding delta.
   * @param delta the delta to add to the hyperplane vector.
   */
  public void addDelta(Vector delta) {
    this.hyperplane = this.hyperplane.plus(delta);
  }
  
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder("Model: ");
    for (int i = 0; i < this.hyperplane.size(); i++) {
      builder.append("  ").append(this.hyperplane.get(i));
    }
    builder.append(" C: ").append(this.bias);
    return builder.toString();
  }
  
  /**
   * Shift the bias of the model.
   * @param factor factor to multiply the bias by.
   */
  public synchronized void shiftBias(double factor) {
    this.bias += factor;
  }
  
  /**
   * Multiply the weight at index by delta.
   * @param index the index of the element to update.
   * @param delta the delta to multiply the element with.
   */
  public void timesDelta(int index, double delta) {
    double element = this.hyperplane.get(index);
    element *= delta;
    this.hyperplane.setQuick(index, element);
  }
}
