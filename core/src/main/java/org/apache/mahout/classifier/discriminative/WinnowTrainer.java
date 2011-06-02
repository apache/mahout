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

import java.util.Iterator;

import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class implements training according to the winnow update algorithm.
 */
public class WinnowTrainer extends LinearTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(WinnowTrainer.class);
  
  /** Promotion step to multiply weights with on update. */
  private final double promotionStep;
  
  public WinnowTrainer(int dimension, double promotionStep, double threshold, double init, double initBias) {
    super(dimension, threshold, init, initBias);
    this.promotionStep = promotionStep;
  }
  
  public WinnowTrainer(int dimension, double promotionStep) {
    this(dimension, promotionStep, 0.5, 1, 0);
  }
  
  /**
   * Initializes with dimension and promotionStep of 2.
   * 
   * @param dimension
   *          number of features.
   */
  public WinnowTrainer(int dimension) {
    this(dimension, 2);
  }
  
  /**
   * {@inheritDoc} Winnow update works such that in case the predicted label
   * does not match the real label, the weight vector is updated as follows: In
   * case the prediction was positiv but should have been negative, all entries
   * in the weight vector that correspond to non null features in the example
   * are doubled.
   * 
   * In case the prediction was negative but should have been positive, all
   * entries in the weight vector that correspond to non null features in the
   * example are halfed.
   */
  @Override
  protected void update(double label, Vector dataPoint, LinearModel model) {
    if (label > 0) {
      // case one
      Vector updateVector = dataPoint.times(1 / this.promotionStep);
      log.info("Winnow update positive: {}", updateVector);
      Iterator<Vector.Element> iter = updateVector.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element element = iter.next();
        model.timesDelta(element.index(), element.get());
      }
    } else {
      // case two
      Vector updateVector = dataPoint.times(1 / this.promotionStep);
      log.info("Winnow update negative: {}", updateVector);
      Iterator<Vector.Element> iter = updateVector.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element element = iter.next();
        model.timesDelta(element.index(), element.get());
      }
    }
    log.info(model.toString());
  }
}
