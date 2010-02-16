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

public class HybridCache<K,V> implements Cache<K,V> {
  
  private final int lfuCapacity;
  
  private final int lruCapacity;
  
  private final LRUCache<K,V> lruCache;
  
  private final LFUCache<K,V> lfuCache;
  
  public HybridCache(int lfuCapacity, int lruCapacity) {
    
    this.lfuCapacity = lfuCapacity;
    this.lruCapacity = lruCapacity;
    
    lruCache = new LRUCache<K,V>(lruCapacity);
    lfuCache = new LFUCache<K,V>(lfuCapacity);
    
  }
  
  @Override
  public long capacity() {
    return lfuCapacity + lruCapacity;
  }
  
  @Override
  public V get(K key) {
    V LRUObject = LRUGet(key);
    if (LRUObject != null) {
      return LRUObject;
    }
    
    V lFUObject = LFUGet(key);
    if (lFUObject != null) {
      return lFUObject;
    }
    
    return null;
  }
  
  private V LFUGet(K key) {
    if (lfuCache.getEvictionCount() >= lfuCapacity) {
      return lfuCache.quickGet(key);
    }
    return lfuCache.get(key);
  }
  
  private V LRUGet(K key) {
    return lruCache.get(key);
  }
  
  @Override
  public void set(K key, V value) {
    
    if (lfuCache.size() < lfuCapacity) {
      lfuCache.set(key, value);
    } else if (lfuCache.getEvictionCount() < lfuCapacity) {
      lfuCache.set(key, value);
      lruCache.set(key, value);
    } else {
      lruCache.set(key, value);
    }
  }
  
  @Override
  public long size() {
    return lfuCache.size() + lruCache.size();
  }
  
  @Override
  public boolean contains(K key) {
    return lruCache.contains(key) || lfuCache.contains(key);
  }
  
}
