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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataUtils;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;

import java.util.Arrays;

/**
 * Optimized implementation of IgSplit<br>
 * This class can be used when the criterion variable is the categorical attribute.
 */
public class OptIgSplit extends IgSplit {

  private int[][] counts;

  private int[] countAll;

  private int[] countLess;

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
    double[] values = data.values(attr);
    int[][] counts = new int[values.length][data.getDataset().nblabels()];
    int[] countAll = new int[data.getDataset().nblabels()];

    Dataset dataset = data.getDataset();

    // compute frequencies
    for (int index = 0; index < data.size(); index++) {
      Instance instance = data.get(index);
      counts[ArrayUtils.indexOf(values, instance.get(attr))][(int) dataset.getLabel(instance)]++;
      countAll[(int) dataset.getLabel(instance)]++;
    }

    int size = data.size();
    double hy = entropy(countAll, size); // H(Y)
    double hyx = 0.0; // H(Y|X)
    double invDataSize = 1.0 / size;

    for (int index = 0; index < values.length; index++) {
      size = DataUtils.sum(counts[index]);
      hyx += size * invDataSize * entropy(counts[index], size);
    }

    double ig = hy - hyx;
    return new Split(attr, ig);
  }

  /**
   * Return the sorted list of distinct values for the given attribute
   */
  private static double[] sortedValues(Data data, int attr) {
    double[] values = data.values(attr);
    Arrays.sort(values);

    return values;
  }

  /**
   * Instantiates the counting arrays
   */
  void initCounts(Data data, double[] values) {
    counts = new int[values.length][data.getDataset().nblabels()];
    countAll = new int[data.getDataset().nblabels()];
    countLess = new int[data.getDataset().nblabels()];
  }

  void computeFrequencies(Data data, int attr, double[] values) {
    Dataset dataset = data.getDataset();

    for (int index = 0; index < data.size(); index++) {
      Instance instance = data.get(index);
      counts[ArrayUtils.indexOf(values, instance.get(attr))][(int) dataset.getLabel(instance)]++;
      countAll[(int) dataset.getLabel(instance)]++;
    }
  }

  /**
   * Computes the best split for a NUMERICAL attribute
   */
  Split numericalSplit(Data data, int attr) {
    double[] values = sortedValues(data, attr);

    initCounts(data, values);

    computeFrequencies(data, attr, values);

    int size = data.size();
    double hy = entropy(countAll, size);
    double invDataSize = 1.0 / size;

    int best = -1;
    double bestIg = -1.0;

    // try each possible split value
    for (int index = 0; index < values.length; index++) {
      double ig = hy;

      // instance with attribute value < values[index]
      size = DataUtils.sum(countLess);
      ig -= size * invDataSize * entropy(countLess, size);

      // instance with attribute value >= values[index]
      size = DataUtils.sum(countAll);
      ig -= size * invDataSize * entropy(countAll, size);

      if (ig > bestIg) {
        bestIg = ig;
        best = index;
      }

      DataUtils.add(countLess, counts[index]);
      DataUtils.dec(countAll, counts[index]);
    }

    if (best == -1) {
      throw new IllegalStateException("no best split found !");
    }
    return new Split(attr, bestIg, values[best]);
  }

  /**
   * Computes the Entropy
   *
   * @param counts   counts[i] = numInstances with label i
   * @param dataSize numInstances
   */
  private static double entropy(int[] counts, int dataSize) {
    if (dataSize == 0) {
      return 0.0;
    }

    double entropy = 0.0;
    double invDataSize = 1.0 / dataSize;

    for (int count : counts) {
      if (count == 0) {
        continue; // otherwise we get a NaN
      }
      double p = count * invDataSize;
      entropy += -p * Math.log(p) / LOG2;
    }

    return entropy;
  }

}
