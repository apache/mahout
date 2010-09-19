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

package org.apache.mahout.classifier.sgd;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.Functions;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Uses sample data to reverse engineer a feature-hashed model.
 *
 * The result gives approximate weights for features and interactions
 * in the original space.
 *
 * The idea is that the hashed encoders have the option of having a trace dictionary.  This
 * tells us where each feature is hashed to, or each feature/value combination in the case
 * of word-like values.  Using this dictionary, we can put values into a synthetic feature
 * vector in just the locations specified by a single feature or interaction.  Then we can
 * push this through a linear part of a model to see the contribution of that input. For
 * any generalized linear model like logistic regression, there is a linear part of the
 * model that allows this.
 *
 * What the ModelDissector does is to accept a trace dictionary and a model in an update
 * method.  It figures out the weights for the elements in the trace dictionary and stashes
 * them.  Then in a summary method, the biggest weights are returned.  This update/flush
 * style is used so that the trace dictionary doesn't have to grow to enormous levels,
 * but instead can be cleared between updates.
 */
public class ModelDissector {
  private Map<String,Vector> weightMap;

  public ModelDissector(int n) {
    weightMap = Maps.newHashMap();
  }

  public void update(Vector features, Map<String, Set<Integer>> traceDictionary, AbstractVectorClassifier learner) {
    features.assign(0);
    for (String feature : traceDictionary.keySet()) {
      if (!weightMap.containsKey(feature)) {
        for (Integer where : traceDictionary.get(feature)) {
          features.set(where, 1);
        }

        Vector v = learner.classifyNoLink(features);
        weightMap.put(feature, v);

        for (Integer where : traceDictionary.get(feature)) {
          features.set(where, 0);
        }
      }
    }

  }

  public List<Weight> summary(int n) {
    PriorityQueue<Weight> pq = new PriorityQueue<Weight>();
    for (String s : weightMap.keySet()) {
      pq.add(new Weight(s, weightMap.get(s)));
      while (pq.size() > n) {
        pq.poll();
      }
    }
    List<Weight> r = Lists.newArrayList(pq);
    Collections.sort(r, Ordering.natural().reverse());
    return r;
  }

  public static class Weight implements Comparable<Weight> {
    private String feature;
    private double value;
    private int maxIndex;

    public Weight(String feature, Vector weights) {
      this.feature = feature;
      // pick out the weight with the largest abs value, but don't forget the sign
      value = weights.aggregate(new BinaryFunction() {
        @Override
        public double apply(double arg1, double arg2) {
          int r = Double.compare(Math.abs(arg1), Math.abs(arg2));
          if (r < 0) {
            return arg2;
          } else if (r > 0) {
            return arg1;
          } else {
            return Math.max(arg1, arg2);
          }
        }
      }, Functions.IDENTITY);
      maxIndex = weights.maxValueIndex();
    }

    @Override
    public int compareTo(Weight other) {
      int r = Double.compare(Math.abs(this.value), Math.abs(other.value));
      if (r != 0) {
        return r;
      } else {
        return feature.compareTo(other.feature);
      }
    }

    public String getFeature() {
      return feature;
    }

    public double getWeight() {
      return value;
    }

    public int getMaxImpact() {
      return maxIndex;
    }
  }
}
