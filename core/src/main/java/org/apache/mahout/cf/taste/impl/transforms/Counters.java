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

package org.apache.mahout.cf.taste.impl.transforms;

import org.apache.mahout.cf.taste.impl.common.FastMap;

import java.util.Map;

/**
 * <p>A simple, fast utility class that maps keys to counts.</p>
 */
final class Counters<T> {

  private final Map<T, MutableInteger> counts = new FastMap<T, MutableInteger>();

  void increment(T key) {
    MutableInteger count = counts.get(key);
    if (count == null) {
      MutableInteger newCount = new MutableInteger();
      newCount.value = 1;
      counts.put(key, newCount);
    } else {
      count.value++;
    }
  }

  int getCount(T key) {
    MutableInteger count = counts.get(key);
    return count == null ? 0 : count.value;
  }

  int size() {
    return counts.size();
  }

  Iterable<Map.Entry<T, MutableInteger>> getEntrySet() {
    return counts.entrySet();
  }

  @Override
  public String toString() {
    return "Counters[" + counts + ']';
  }

  static final class MutableInteger {

    // This is intentionally package-private in order to allow access from the containing Counters class
    // without making the compiler generate a synthetic accessor
    int value;

    @Override
    public String toString() {
      return "MutableInteger[" + value + ']';
    }
  }

}
