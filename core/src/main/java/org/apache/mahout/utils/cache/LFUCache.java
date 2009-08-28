package org.apache.mahout.utils.cache;

import java.util.*;
import org.apache.mahout.common.Pair;


public class LFUCache<K, V> implements org.apache.mahout.utils.Cache<K, V> {

  SortedMap<Long, Set<K>> evictionMap = null;

  Map<K, Pair<V, Long>> dataMap = null;

  int capacity = 0;

  int evictionCount = 0;

  public LFUCache(int capacity) {

    this.capacity = capacity;

    evictionMap = new TreeMap<Long, Set<K>>();
    dataMap = new HashMap<K, Pair<V, Long>>(capacity);

  }

  @Override
  public long capacity() {
    return capacity;
  }

  public int getEvictionCount() {
    return this.evictionCount;
  }

  @Override
  public V get(K key) {
    Pair<V,Long> data = dataMap.get(key);
    if (data == null)
      return null;
    else {
      V value = data.left();
      long count = data.right();
      data.setValue(Long.valueOf(count + 1));
      incrementHit(key, Long.valueOf(count));
      return value;
    }

  }
  
  public V quickGet(K key){
    Pair<V,Long> data = dataMap.get(key);
    if (data == null)
      return null;
    else
      return data.left();
  }

  private void incrementHit(K key, Long count) {
    Set<K> keys = evictionMap.get(count);
    if (keys == null)
      throw new ConcurrentModificationException();
    if (keys.remove(key) == false)
      throw new ConcurrentModificationException();
    if (keys.size() == 0)
      evictionMap.remove(count);
    count = Long.valueOf(count.longValue() + 1);
    Set<K> keysNew = evictionMap.get(count);
    if (keysNew == null) {
      keysNew = new LinkedHashSet<K>();
      evictionMap.put(count, keysNew);
    }
    keysNew.add(key);
  }

  @Override
  public void set(K key, V value) {
    if (dataMap.containsKey(key))
      return;
    if (capacity == dataMap.size()) // Cache Full
    {
      removeLeastFrequent();
    }
    Long count = Long.valueOf(1);
    Pair<V, Long> data = new Pair<V, Long>(value, count);
    dataMap.put(key, data);

    Set<K> keys = evictionMap.get(count);
    if (keys == null) {
      keys = new LinkedHashSet<K>();
      evictionMap.put(count, keys);
    }
    keys.add(key);

  }
  private void removeLeastFrequent() {
    Long key = evictionMap.firstKey();
    Set<K> values = evictionMap.get(key);
    Iterator<K> it = values.iterator();
    K keyToBeRemoved = it.next();
    values.remove(keyToBeRemoved);
    if (values.size() == 0)
      evictionMap.remove(key);
    dataMap.remove(keyToBeRemoved);
    evictionCount++;

  }

  @Override
  public long size() {
    return dataMap.size();
  }

  @Override
  public boolean contains(K key) {
    return (dataMap.containsKey(key));
  }

}
