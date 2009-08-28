package org.apache.mahout.utils.cache;

public class HybridCache<K, V> implements org.apache.mahout.utils.Cache<K, V> {

  private int LFUCapacity = 0;

  int LRUCapacity = 0;

  private LRUCache<K, V> lruCache = null;

  private LFUCache<K, V> lfuCache = null;

  public HybridCache(int lfuCapacity, int lruCapacity) {

    this.LFUCapacity = lfuCapacity;
    this.LRUCapacity = lruCapacity;

    lruCache = new LRUCache<K, V>(LRUCapacity);
    lfuCache = new LFUCache<K, V>(LFUCapacity);

  }

  @Override
  public long capacity() {
    return LFUCapacity + LRUCapacity;
  }

  @Override
  public V get(K key) {
    V LRUObject = LRUGet(key);
    if (LRUObject != null)
      return LRUObject;

    V lFUObject = LFUGet(key);
    if (lFUObject != null)
      return lFUObject;

    return null;
  }

  private V LFUGet(K key) {
    if (lfuCache.getEvictionCount() >= LFUCapacity)
      return lfuCache.quickGet(key);
    return lfuCache.get(key);
  }

  private V LRUGet(K key) {
    return lruCache.get(key);
  }

  @Override
  public void set(K key, V value) {

    if (lfuCache.size() < LFUCapacity)
      lfuCache.set(key, value);
    else if (lfuCache.getEvictionCount() < LFUCapacity) {
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
    if (lruCache.contains(key))
      return true;
    else
      return lfuCache.contains(key);
  }

}
