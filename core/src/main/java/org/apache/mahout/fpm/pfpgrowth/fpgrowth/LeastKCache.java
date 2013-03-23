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

package org.apache.mahout.fpm.pfpgrowth.fpgrowth;

import com.google.common.collect.Maps;

import java.util.Collections;
import java.util.Map;
import java.util.PriorityQueue;

public class LeastKCache<K extends Comparable<? super K>,V> {
  
  private final int capacity;
  private final Map<K,V> cache;
  private final PriorityQueue<K> queue;
  
  public LeastKCache(int capacity) {
    this.capacity = capacity;
    cache = Maps.newHashMapWithExpectedSize(capacity);
    queue = new PriorityQueue<K>(capacity + 1, Collections.reverseOrder());
  }

  public final V get(K key) {
    return cache.get(key);
  }
  
  public final void set(K key, V value) {
    if (!contains(key)) {
      queue.add(key);
    }
    cache.put(key, value);
    while (queue.size() > capacity) {
      K k = queue.poll();
      cache.remove(k);
    }
  }
  
  public final boolean contains(K key) {
    return cache.containsKey(key);
  }
  
}
