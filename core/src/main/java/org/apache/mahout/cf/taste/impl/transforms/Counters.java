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

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;

import java.util.Map;

/** <p>A simple, fast utility class that maps keys to counts.</p> */
final class Counters {

  private final FastByIDMap<int[]> counts = new FastByIDMap<int[]>();

  void increment(long key) {
    int[] count = counts.get(key);
    if (count == null) {
      int[] newCount = new int[1];
      newCount[0] = 1;
      counts.put(key, newCount);
    } else {
      count[0]++;
    }
  }

  int getCount(long key) {
    int[] count = counts.get(key);
    return count == null ? 0 : count[0];
  }

  int size() {
    return counts.size();
  }

  Iterable<Map.Entry<Long, int[]>> getEntrySet() {
    return counts.entrySet();
  }

  @Override
  public String toString() {
    return "Counters[" + counts + ']';
  }

}
