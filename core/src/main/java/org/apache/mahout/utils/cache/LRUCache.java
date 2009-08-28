package org.apache.mahout.utils.cache;

import java.util.LinkedHashMap;
import java.util.Map;


public class LRUCache<K, V> implements org.apache.mahout.utils.Cache<K, V> {

  int capacity = 0;
  
  private  Map<K, V> lruCache = null;
  
  public LRUCache(final int capacity) {

    this.capacity = capacity;

    lruCache = new LinkedHashMap<K,V>( (int)(capacity/0.75f + 1), 0.75f, true) { 
      private static final long serialVersionUID = -576585264027935752L;
      private final int MAX_ENTRIES = capacity;
      @Override protected boolean removeEldestEntry (Map.Entry<K,V> eldest) {
         return size() > MAX_ENTRIES; 
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
