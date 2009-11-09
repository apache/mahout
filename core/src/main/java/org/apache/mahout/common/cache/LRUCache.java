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

import java.util.LinkedHashMap;
import java.util.Map;

public class LRUCache<K, V> implements Cache<K, V> {

  private int capacity = 0;
  
  private Map<K, V> lruCache = null;
  
  public LRUCache(final int capacity) {

    this.capacity = capacity;

    lruCache = new LinkedHashMap<K,V>( (int)(capacity/0.75f + 1), 0.75f, true) { 
      @Override
      protected boolean removeEldestEntry (Map.Entry<K,V> eldest) {
        return size() > capacity;
      }
    };
      
  }

  @Override
  public long capacity() {
    return capacity;
  }

  @Override
  public V get(K key) {
    return lruCache.get(key);
  }

  @Override
  public void set(K key, V value) {
      lruCache.put(key,value);
  }

  @Override
  public long size() {
    return lruCache.size();
  }

  @Override
  public boolean contains(K key) {
    return (lruCache.containsKey(key));  
  }

}
