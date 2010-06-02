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

import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementors of this class need to provide a way to train linear
 * discriminative classifiers.
 * 
 * As this is just the reference implementation we assume that the dataset fits
 * into main memory - this should be the first thing to change when switching to
 * Hadoop.
 */
public abstract class LinearTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(LinearTrainer.class);

  /** The model to train. */
  private final LinearModel model;
  
  /**
   * Initialize the trainer. Distance is initialized to cosine distance, all
   * weights are represented through a dense vector.
   * 
   * 
   * @param dimension
   *          number of expected features.
   * @param threshold
   *          threshold to use for classification.
   * @param init
   *          initial value of weight vector.
   * @param initBias
   *          initial classification bias.
   */
  protected LinearTrainer(int dimension, double threshold,
                          double init, double initBias) {
    DenseVector initialWeights = new DenseVector(dimension);
    initialWeights.assign(init);
    this.model = new LinearModel(initialWeights, initBias, threshold);
  }
  
  /**
   * Initializes training. Runs through all data points in the training set and
   * updates the weight vector whenever a classification error occurs.
   * 
   * Can be called multiple times.
   * 
   * @param dataset
   *          the dataset to train on. Each column is treated as point.
   * @param labelset
   *          the set of labels, one for each data point. If the cardinalities
   *          of data- and labelset do not match, a CardinalityException is
   *          thrown
   */
  public void train(Vector labelset, Matrix dataset) throws TrainingException {
    if (labelset.size() != dataset.size()[1]) {
      throw new CardinalityException(labelset.size(), dataset.size()[1]);
    }
    
    boolean converged = false;
    int iteration = 0;
    while (!converged) {
      if (iteration > 1000) {
        throw new TrainingException("Too many iterations needed to find hyperplane.");
      }
      
      converged = true;
      int columnCount = dataset.size()[1];
      for (int i = 0; i < columnCount; i++) {
        Vector dataPoint = dataset.getColumn(i);
        log.debug("Training point: " + dataPoint);
        
        synchronized (this.model) {
          boolean prediction = model.classify(dataPoint);
          double label = labelset.get(i);
          if (label <= 0 && prediction || label > 0 && !prediction) {
            log.debug("updating");
            converged = false;
            update(label, dataPoint, this.model);
          }
        }
      }
    }
  }
  
  /**
   * Retrieves the trained model if called after train, otherwise the raw model.
   */
  public LinearModel getModel() {
    return this.model;
  }
  
  /**
   * Implement this method to match your training strategy.
   * 
   * @param model
   *          the model to update.
   * @param label
   *          the target label of the wrongly classified data point.
   * @param dataPoint
   *          the data point that was classified incorrectly.
   */
  protected abstract void update(double label, Vector dataPoint, LinearModel model);
  
}
