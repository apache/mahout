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
import org.apache.mahout.classifier.df.data.Instance;

import java.util.Arrays;

/**
 * Regression problem implementation of IgSplit.
 * This class can be used when the criterion variable is the numerical attribute.
 */
public class RegressionSplit extends IgSplit {
  
  /**
   * Comparator for Instance sort
   */
  private static class InstanceComparator implements java.util.Comparator<Instance> {
    private final int attr;

    InstanceComparator(int attr) {
      this.attr = attr;
    }
    
    @Override
    public int compare(Instance arg0, Instance arg1) {
      return Double.compare(arg0.get(attr), arg1.get(attr));
    }
  }
  
  @Override
  public Split computeSplit(Data data, int attr) {
    if (data.getDataset().isNumerical(attr)) {
      return numericalSplit(data, attr);
    } else {
      return categoricalSplit(data, attr);
    }
  }
  
  /**
   * Computes the split for a CATEGORICAL attribute
   */
  private static Split categoricalSplit(Data data, int attr) {
    double[] sums = new double[data.getDataset().nbValues(attr)];
    double[] sumSquared = new double[data.getDataset().nbValues(attr)];
    double[] counts = new double[data.getDataset().nbValues(attr)];
    double totalSum = 0;
    double totalSumSquared = 0;

    // sum and sum of squares
    for (int i = 0; i < data.size(); i++) {
      Instance instance = data.get(i);
      int value = (int) instance.get(attr);
      double label = data.getDataset().getLabel(instance);
      double square = label * label;

      sums[value] += label;
      sumSquared[value] += square;
      counts[value]++;
      totalSum += label;
      totalSumSquared += square;
    }
    
    // computes the variance
    double totalVar = totalSumSquared - (totalSum * totalSum) / data.size();
    double var = variance(sums, sumSquared, counts);
    double ig = totalVar - var;

    return new Split(attr, ig);
  }
  
  /**
   * Computes the best split for a NUMERICAL attribute
   */
  static Split numericalSplit(Data data, int attr) {

    // Instance sort
    Instance[] instances = new Instance[data.size()];
    for (int i = 0; i < data.size(); i++) {
      instances[i] = data.get(i);
    }
    Arrays.sort(instances, new InstanceComparator(attr));

    // sum and sum of squares
    double totalSum = 0.0;
    double totalSumSquared = 0.0;
    for (Instance instance : instances) {
      double label = data.getDataset().getLabel(instance);
      totalSum += label;
      totalSumSquared += label * label;
    }
    double[] sums = new double[2];
    double[] curSums = new double[2];
    sums[1] = curSums[1] = totalSum;
    double[] sumSquared = new double[2];
    double[] curSumSquared = new double[2];
    sumSquared[1] = curSumSquared[1] = totalSumSquared;
    double[] counts = new double[2];
    double[] curCounts = new double[2];
    counts[1] = curCounts[1] = data.size();

    // find the best split point
    double curSplit = instances[0].get(attr);
    double bestVal = Double.MAX_VALUE;
    double split = Double.NaN;
    for (Instance instance : instances) {
      if (instance.get(attr) > curSplit) {
        double curVal = variance(curSums, curSumSquared, curCounts);
        if (curVal < bestVal) {
          bestVal = curVal;
          split = (instance.get(attr) + curSplit) / 2.0;
          for (int j = 0; j < 2; j++) {
            sums[j] = curSums[j];
            sumSquared[j] = curSumSquared[j];
            counts[j] = curCounts[j];
          }
        }
      }

      curSplit = instance.get(attr);

      double label = data.getDataset().getLabel(instance);
      double square = label * label;

      curSums[0] += label;
      curSumSquared[0] += square;
      curCounts[0]++;

      curSums[1] -= label;
      curSumSquared[1] -= square;
      curCounts[1]--;
    }

    // computes the variance
    double totalVar = totalSumSquared - (totalSum * totalSum) / data.size();
    double var = variance(sums, sumSquared, counts);
    double ig = totalVar - var;

    return new Split(attr, ig, split);
  }
  
  /**
   * Computes the variance
   * 
   * @param s
   *          data
   * @param ss
   *          squared data
   * @param dataSize
   *          numInstances
   */
  private static double variance(double[] s, double[] ss, double[] dataSize) {
    double var = 0;
    for (int i = 0; i < s.length; i++) {
      if (dataSize[i] > 0) {
        var += ss[i] - ((s[i] * s[i]) / dataSize[i]);
      }
    }
    return var;
  }
}
