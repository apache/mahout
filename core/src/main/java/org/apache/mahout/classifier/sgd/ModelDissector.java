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
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
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
  private final Map<String,Vector> weightMap;

  public ModelDissector() {
    weightMap = Maps.newHashMap();
  }

  /**
   * Probes a model to determine the effect of a particular variable.  This is done
   * with the ade of a trace dictionary which has recorded the locations in the feature
   * vector that are modified by various variable values.  We can set these locations to
   * 1 and then look at the resulting score.  This tells us the weight the model places
   * on that variable.
   * @param features               A feature vector to use (destructively)
   * @param traceDictionary        A trace dictionary containing variables and what locations
   *                               in the feature vector are affected by them
   * @param learner                The model that we are probing to find weights on features
   */

  public void update(Vector features, Map<String, Set<Integer>> traceDictionary, AbstractVectorClassifier learner) {
    // zero out feature vector
    features.assign(0);
    for (Map.Entry<String, Set<Integer>> entry : traceDictionary.entrySet()) {
      // get a feature and locations where it is stored in the feature vector
      String key = entry.getKey();
      Set<Integer> value = entry.getValue();

      // if we haven't looked at this feature yet
      if (!weightMap.containsKey(key)) {
        // put probe values in the feature vector
        for (Integer where : value) {
          features.set(where, 1);
        }

        // see what the model says
        Vector v = learner.classifyNoLink(features);
        weightMap.put(key, v);

        // and zero out those locations again
        for (Integer where : value) {
          features.set(where, 0);
        }
      }
    }
  }

  /**
   * Returns the n most important features with their
   * weights, most important category and the top few
   * categories that they affect.
   * @param n      How many results to return.
   * @return       A list of the top variables.
   */
  public List<Weight> summary(int n) {
    Queue<Weight> pq = new PriorityQueue<Weight>();
    for (Map.Entry<String, Vector> entry : weightMap.entrySet()) {
      pq.add(new Weight(entry.getKey(), entry.getValue()));
      while (pq.size() > n) {
        pq.poll();
      }
    }
    List<Weight> r = Lists.newArrayList(pq);
    Collections.sort(r, Ordering.natural().reverse());
    return r;
  }

  private static final class Category implements Comparable<Category> {
    private final int index;
    private final double weight;

    private Category(int index, double weight) {
      this.index = index;
      this.weight = weight;
    }

    @Override
    public int compareTo(Category o) {
      int r = Double.compare(Math.abs(weight), Math.abs(o.weight));
      if (r == 0) {
        if (o.index < index) {
          return -1;
        }
        if (o.index > index) {
          return 1;
        }
        return 0;
      }
      return r;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof Category)) {
        return false;
      }
      Category other = (Category) o;
      return index == other.index && weight == other.weight;
    }

    @Override
    public int hashCode() {
      return RandomUtils.hashDouble(weight) ^ index;
    }

  }

  public static class Weight implements Comparable<Weight> {
    private final String feature;
    private final double value;
    private final int maxIndex;
    private final List<Category> categories;

    public Weight(String feature, Vector weights) {
      this(feature, weights, 3);
    }

    public Weight(String feature, Vector weights, int n) {
      this.feature = feature;
      // pick out the weight with the largest abs value, but don't forget the sign
      Queue<Category> biggest = new PriorityQueue<Category>(n + 1, Ordering.natural());
      for (Vector.Element element : weights.all()) {
        biggest.add(new Category(element.index(), element.get()));
        while (biggest.size() > n) {
          biggest.poll();
        }
      }
      categories = Lists.newArrayList(biggest);
      Collections.sort(categories, Ordering.natural().reverse());
      value = categories.get(0).weight;
      maxIndex = categories.get(0).index;
    }

    @Override
    public int compareTo(Weight other) {
      int r = Double.compare(Math.abs(this.value), Math.abs(other.value));
      if (r == 0) {
        return feature.compareTo(other.feature);
      }
      return r;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof Weight)) {
        return false;
      }
      Weight other = (Weight) o;
      return feature.equals(other.feature)
          && value == other.value
          && maxIndex == other.maxIndex
          && categories.equals(other.categories);
    }

    @Override
    public int hashCode() {
      return feature.hashCode() ^ RandomUtils.hashDouble(value) ^ maxIndex ^ categories.hashCode();
    }

    public String getFeature() {
      return feature;
    }

    public double getWeight() {
      return value;
    }

    public double getWeight(int n) {
      return categories.get(n).weight;
    }

    public double getCategory(int n) {
      return categories.get(n).index;
    }

    public int getMaxImpact() {
      return maxIndex;
    }
  }
}
