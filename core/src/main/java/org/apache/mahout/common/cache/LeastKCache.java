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

package org.apache.mahout.common.cache;

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class LeastKCache<K extends Comparable<? super K>,V> implements Cache<K,V> {
  
  private final int capacity;
  
  private final Map<K,V> cache;
  
  private final PriorityQueue<K> queue;
  
  public LeastKCache(int capacity) {
    
    this.capacity = capacity;
    
    cache = new HashMap<K,V>(capacity);
    queue = new PriorityQueue<K>(capacity, new Comparator<K>() {
      
      @Override
      public int compare(K o1, K o2) {
        return o2.compareTo(o1);
      }
      
    });
    
  }
  
  @Override
  public final long capacity() {
    return capacity;
  }
  
  @Override
  public final V get(K key) {
    return cache.get(key);
  }
  
  @Override
  public final void set(K key, V value) {
    if (contains(key) == false) {
      queue.add(key);
    }
    cache.put(key, value);
    while (queue.size() > capacity) {
      K k = queue.poll();
      cache.remove(k);
    }
  }
  
  @Override
  public final long size() {
    return cache.size();
  }
  
  @Override
  public final boolean contains(K key) {
    return cache.containsKey(key);
  }
  
}
