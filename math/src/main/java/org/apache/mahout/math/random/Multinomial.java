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

package org.apache.mahout.math.random;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.list.DoubleArrayList;

/**
 * Multinomial sampler that allows updates to element probabilities.  The basic idea is that sampling is
 * done by using a simple balanced tree.  Probabilities are kept in the tree so that we can navigate to
 * any leaf in log N time.  Updates are simple because we can just propagate them upwards.
 * <p/>
 * In order to facilitate access by value, we maintain an additional map from value to tree node.
 */
public final class Multinomial<T> implements Sampler<T>, Iterable<T> {
  // these lists use heap ordering.  Thus, the root is at location 1, first level children at 2 and 3, second level
  // at 4, 5 and 6, 7.
  private final DoubleArrayList weight = new DoubleArrayList();
  private final List<T> values = Lists.newArrayList();
  private final Map<T, Integer> items = Maps.newHashMap();
  private Random rand = RandomUtils.getRandom();

  public Multinomial() {
    weight.add(0);
    values.add(null);
  }

  public Multinomial(Multiset<T> counts) {
    this();
    Preconditions.checkArgument(!counts.isEmpty(), "Need some data to build sampler");
    rand = RandomUtils.getRandom();
    for (T t : counts.elementSet()) {
      add(t, counts.count(t));
    }
  }

  public Multinomial(Iterable<WeightedThing<T>> things) {
    this();
    for (WeightedThing<T> thing : things) {
      add(thing.getValue(), thing.getWeight());
    }
  }

  public void add(T value, double w) {
    Preconditions.checkNotNull(value);
    Preconditions.checkArgument(!items.containsKey(value));

    int n = this.weight.size();
    if (n == 1) {
      weight.add(w);
      values.add(value);
      items.put(value, 1);
    } else {
      // parent comes down
      weight.add(weight.get(n / 2));
      values.add(values.get(n / 2));
      items.put(values.get(n / 2), n);
      n++;

      // new item goes in
      items.put(value, n);
      this.weight.add(w);
      values.add(value);

      // parents get incremented all the way to the root
      while (n > 1) {
        n /= 2;
        this.weight.set(n, this.weight.get(n) + w);
      }
    }
  }

  public double getWeight(T value) {
    if (items.containsKey(value)) {
      return weight.get(items.get(value));
    } else {
      return 0;
    }
  }

  public double getProbability(T value) {
    if (items.containsKey(value)) {
      return weight.get(items.get(value)) / weight.get(1);
    } else {
      return 0;
    }
  }

  public double getWeight() {
    if (weight.size() > 1) {
      return weight.get(1);
    } else {
      return 0;
    }
  }

  public void delete(T value) {
    set(value, 0);
  }

  public void set(T value, double newP) {
    Preconditions.checkArgument(items.containsKey(value));
    int n = items.get(value);
    if (newP <= 0) {
      // this makes the iterator not see such an element even though we leave a phantom in the tree
      // Leaving the phantom behind simplifies tree maintenance and testing, but isn't really necessary.
      items.remove(value);
    }
    double oldP = weight.get(n);
    while (n > 0) {
      weight.set(n, weight.get(n) - oldP + newP);
      n /= 2;
    }
  }

  @Override
  public T sample() {
    Preconditions.checkArgument(!weight.isEmpty());
    return sample(rand.nextDouble());
  }

  public T sample(double u) {
    u *= weight.get(1);

    int n = 1;
    while (2 * n < weight.size()) {
      // children are at 2n and 2n+1
      double left = weight.get(2 * n);
      if (u <= left) {
        n = 2 * n;
      } else {
        u -= left;
        n = 2 * n + 1;
      }
    }
    return values.get(n);
  }

  /**
   * Exposed for testing only.  Returns a list of the leaf weights.  These are in an
   * order such that probing just before and after the cumulative sum of these weights
   * will touch every element of the tree twice and thus will make it possible to test
   * every possible left/right decision in navigating the tree.
   */
  List<Double> getWeights() {
    List<Double> r = Lists.newArrayList();
    int i = Integer.highestOneBit(weight.size());
    while (i < weight.size()) {
      r.add(weight.get(i));
      i++;
    }
    i /= 2;
    while (i < Integer.highestOneBit(weight.size())) {
      r.add(weight.get(i));
      i++;
    }
    return r;
  }

  @Override
  public Iterator<T> iterator() {
    return new AbstractIterator<T>() {
      Iterator<T> valuesIterator = Iterables.skip(values, 1).iterator();
      @Override
      protected T computeNext() {
        while (valuesIterator.hasNext()) {
          T next = valuesIterator.next();
          if (items.containsKey(next)) {
            return next;
          }
        }
        return endOfData();
      }
    };
  }
}
