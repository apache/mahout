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

package org.apache.mahout.classifier.df.split;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataUtils;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * <p>Optimized implementation of IgSplit.
 * This class can be used when the criterion variable is the categorical attribute.</p>
 *
 * <p>This code was changed in MAHOUT-1419 to deal in sampled splits among numeric
 * features to fix a performance problem. To generate some synthetic data that exercises
 * the issue, try for example generating 4 features of Normal(0,1) values with a random
 * boolean 0/1 categorical feature. In Scala:</p>
 *
 * {@code
 *  val r = new scala.util.Random()
 *  val pw = new java.io.PrintWriter("random.csv")
 *  (1 to 10000000).foreach(e =>
 *    pw.println(r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               r.nextDouble() + "," +
 *               (if (r.nextBoolean()) 1 else 0))
 *   )
 *   pw.close()
 * }
 */
@Deprecated
public class OptIgSplit extends IgSplit {

  private static final int MAX_NUMERIC_SPLITS = 16;

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
    double[] values = data.values(attr).clone();

    double[] splitPoints = chooseCategoricalSplitPoints(values);

    int numLabels = data.getDataset().nblabels();
    int[][] counts = new int[splitPoints.length][numLabels];
    int[] countAll = new int[numLabels];

    computeFrequencies(data, attr, splitPoints, counts, countAll);

    int size = data.size();
    double hy = entropy(countAll, size); // H(Y)
    double hyx = 0.0; // H(Y|X)
    double invDataSize = 1.0 / size;

    for (int index = 0; index < splitPoints.length; index++) {
      size = DataUtils.sum(counts[index]);
      hyx += size * invDataSize * entropy(counts[index], size);
    }

    double ig = hy - hyx;
    return new Split(attr, ig);
  }

  static void computeFrequencies(Data data,
                                 int attr,
                                 double[] splitPoints,
                                 int[][] counts,
                                 int[] countAll) {
    Dataset dataset = data.getDataset();

    for (int index = 0; index < data.size(); index++) {
      Instance instance = data.get(index);
      int label = (int) dataset.getLabel(instance);
      double value = instance.get(attr);
      int split = 0;
      while (split < splitPoints.length && value > splitPoints[split]) {
        split++;
      }
      if (split < splitPoints.length) {
        counts[split][label]++;
      } // Otherwise it's in the last split, which we don't need to count
      countAll[label]++;
    }
  }

  /**
   * Computes the best split for a NUMERICAL attribute
   */
  static Split numericalSplit(Data data, int attr) {
    double[] values = data.values(attr).clone();
    Arrays.sort(values);

    double[] splitPoints = chooseNumericSplitPoints(values);

    int numLabels = data.getDataset().nblabels();
    int[][] counts = new int[splitPoints.length][numLabels];
    int[] countAll = new int[numLabels];
    int[] countLess = new int[numLabels];

    computeFrequencies(data, attr, splitPoints, counts, countAll);

    int size = data.size();
    double hy = entropy(countAll, size);
    double invDataSize = 1.0 / size;

    int best = -1;
    double bestIg = -1.0;

    // try each possible split value
    for (int index = 0; index < splitPoints.length; index++) {
      double ig = hy;

      DataUtils.add(countLess, counts[index]);
      DataUtils.dec(countAll, counts[index]);

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
    }

    if (best == -1) {
      throw new IllegalStateException("no best split found !");
    }
    return new Split(attr, bestIg, splitPoints[best]);
  }

  /**
   * @return an array of values to split the numeric feature's values on when
   *  building candidate splits. When input size is <= MAX_NUMERIC_SPLITS + 1, it will
   *  return the averages between success values as split points. When larger, it will
   *  return MAX_NUMERIC_SPLITS approximate percentiles through the data.
   */
  private static double[] chooseNumericSplitPoints(double[] values) {
    if (values.length <= 1) {
      return values;
    }
    if (values.length <= MAX_NUMERIC_SPLITS + 1) {
      double[] splitPoints = new double[values.length - 1];
      for (int i = 1; i < values.length; i++) {
        splitPoints[i-1] = (values[i] + values[i-1]) / 2.0;
      }
      return splitPoints;
    }
    Percentile distribution = new Percentile();
    distribution.setData(values);
    double[] percentiles = new double[MAX_NUMERIC_SPLITS];
    for (int i = 0 ; i < percentiles.length; i++) {
      double p = 100.0 * ((i + 1.0) / (MAX_NUMERIC_SPLITS + 1.0));
      percentiles[i] = distribution.evaluate(p);
    }
    return percentiles;
  }

  private static double[] chooseCategoricalSplitPoints(double[] values) {
    // There is no great reason to believe that categorical value order matters,
    // but the original code worked this way, and it's not terrible in the absence
    // of more sophisticated analysis
    Collection<Double> uniqueOrderedCategories = new TreeSet<Double>();
    for (double v : values) {
      uniqueOrderedCategories.add(v);
    }
    double[] uniqueValues = new double[uniqueOrderedCategories.size()];
    Iterator<Double> it = uniqueOrderedCategories.iterator();
    for (int i = 0; i < uniqueValues.length; i++) {
      uniqueValues[i] = it.next();
    }
    return uniqueValues;
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

    for (int count : counts) {
      if (count > 0) {
        double p = count / (double) dataSize;
        entropy -= p * Math.log(p);
      }
    }

    return entropy / LOG2;
  }

}
