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

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Instance;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

/**
 * Regression problem implementation of IgSplit. This class can be used when the criterion variable is the numerical
 * attribute.
 */
public class RegressionSplit extends IgSplit {
  
  /**
   * Comparator for Instance sort
   */
  private static class InstanceComparator implements Comparator<Instance>, Serializable {
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
    FullRunningAverage[] ra = new FullRunningAverage[data.getDataset().nbValues(attr)];
    double[] sk = new double[data.getDataset().nbValues(attr)];
    for (int i = 0; i < ra.length; i++) {
      ra[i] = new FullRunningAverage();
    }
    FullRunningAverage totalRa = new FullRunningAverage();
    double totalSk = 0.0;

    for (int i = 0; i < data.size(); i++) {
      // computes the variance
      Instance instance = data.get(i);
      int value = (int) instance.get(attr);
      double xk = data.getDataset().getLabel(instance);
      if (ra[value].getCount() == 0) {
        ra[value].addDatum(xk);
        sk[value] = 0.0;
      } else {
        double mk = ra[value].getAverage();
        ra[value].addDatum(xk);
        sk[value] += (xk - mk) * (xk - ra[value].getAverage());
      }

      // total variance
      if (i == 0) {
        totalRa.addDatum(xk);
        totalSk = 0.0;
      } else {
        double mk = totalRa.getAverage();
        totalRa.addDatum(xk);
        totalSk += (xk - mk) * (xk - totalRa.getAverage());
      }
    }

    // computes the variance gain
    double ig = totalSk;
    for (double aSk : sk) {
      ig -= aSk;
    }

    return new Split(attr, ig);
  }
  
  /**
   * Computes the best split for a NUMERICAL attribute
   */
  private static Split numericalSplit(Data data, int attr) {
    FullRunningAverage[] ra = new FullRunningAverage[2];
    for (int i = 0; i < ra.length; i++) {
      ra[i] = new FullRunningAverage();
    }

    // Instance sort
    Instance[] instances = new Instance[data.size()];
    for (int i = 0; i < data.size(); i++) {
      instances[i] = data.get(i);
    }
    Arrays.sort(instances, new InstanceComparator(attr));

    double[] sk = new double[2];
    for (Instance instance : instances) {
      double xk = data.getDataset().getLabel(instance);
      if (ra[1].getCount() == 0) {
        ra[1].addDatum(xk);
        sk[1] = 0.0;
      } else {
        double mk = ra[1].getAverage();
        ra[1].addDatum(xk);
        sk[1] += (xk - mk) * (xk - ra[1].getAverage());
      }
    }
    double totalSk = sk[1];

    // find the best split point
    double split = Double.NaN;
    double preSplit = Double.NaN;
    double bestVal = Double.MAX_VALUE;
    double bestSk = 0.0;

    // computes total variance
    for (Instance instance : instances) {
      double xk = data.getDataset().getLabel(instance);

      if (instance.get(attr) > preSplit) {
        double curVal = sk[0] / ra[0].getCount() + sk[1] / ra[1].getCount();
        if (curVal < bestVal) {
          bestVal = curVal;
          bestSk = sk[0] + sk[1];
          split = (instance.get(attr) + preSplit) / 2.0;
        }
      }

      // computes the variance
      if (ra[0].getCount() == 0) {
        ra[0].addDatum(xk);
        sk[0] = 0.0;
      } else {
        double mk = ra[0].getAverage();
        ra[0].addDatum(xk);
        sk[0] += (xk - mk) * (xk - ra[0].getAverage());
      }

      double mk = ra[1].getAverage();
      ra[1].removeDatum(xk);
      sk[1] -= (xk - mk) * (xk - ra[1].getAverage());

      preSplit = instance.get(attr);
    }

    // computes the variance gain
    double ig = totalSk - bestSk;

    return new Split(attr, ig, split);
  }
}
