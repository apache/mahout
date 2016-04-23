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

package org.apache.mahout.classifier.df.split;

import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.conditions.Condition;

import java.util.Arrays;

/**
 * Default, not optimized, implementation of IgSplit
 */
@Deprecated
public class DefaultIgSplit extends IgSplit {
  
  /** used by entropy() */
  private int[] counts;
  
  @Override
  public Split computeSplit(Data data, int attr) {
    if (data.getDataset().isNumerical(attr)) {
      double[] values = data.values(attr);
      double bestIg = -1;
      double bestSplit = 0.0;
      
      for (double value : values) {
        double ig = numericalIg(data, attr, value);
        if (ig > bestIg) {
          bestIg = ig;
          bestSplit = value;
        }
      }
      
      return new Split(attr, bestIg, bestSplit);
    } else {
      double ig = categoricalIg(data, attr);
      
      return new Split(attr, ig);
    }
  }
  
  /**
   * Computes the Information Gain for a CATEGORICAL attribute
   */
  double categoricalIg(Data data, int attr) {
    double[] values = data.values(attr);
    double hy = entropy(data); // H(Y)
    double hyx = 0.0; // H(Y|X)
    double invDataSize = 1.0 / data.size();
    
    for (double value : values) {
      Data subset = data.subset(Condition.equals(attr, value));
      hyx += subset.size() * invDataSize * entropy(subset);
    }
    
    return hy - hyx;
  }
  
  /**
   * Computes the Information Gain for a NUMERICAL attribute given a splitting value
   */
  double numericalIg(Data data, int attr, double split) {
    double hy = entropy(data);
    double invDataSize = 1.0 / data.size();
    
    // LO subset
    Data subset = data.subset(Condition.lesser(attr, split));
    hy -= subset.size() * invDataSize * entropy(subset);
    
    // HI subset
    subset = data.subset(Condition.greaterOrEquals(attr, split));
    hy -= subset.size() * invDataSize * entropy(subset);
    
    return hy;
  }
  
  /**
   * Computes the Entropy
   */
  protected double entropy(Data data) {
    double invDataSize = 1.0 / data.size();
    
    if (counts == null) {
      counts = new int[data.getDataset().nblabels()];
    }
    
    Arrays.fill(counts, 0);
    data.countLabels(counts);
    
    double entropy = 0.0;
    for (int label = 0; label < data.getDataset().nblabels(); label++) {
      int count = counts[label];
      if (count == 0) {
        continue; // otherwise we get a NaN
      }
      double p = count * invDataSize;
      entropy += -p * Math.log(p) / LOG2;
    }
    
    return entropy;
  }
  
}
