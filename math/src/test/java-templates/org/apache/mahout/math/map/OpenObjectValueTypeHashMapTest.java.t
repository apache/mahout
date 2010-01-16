/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#if (${valueTypeFloating} == 'true')
#set ($valueEpsilon = ", (${valueType})0.000001")
#else 
#set ($valueEpsilon = "")
#end
  
package org.apache.mahout.math.map;
 
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.math.function.Object${valueTypeCap}Procedure;
import org.apache.mahout.math.function.ObjectProcedure;
import org.apache.mahout.math.list.${valueTypeCap}ArrayList;
import org.apache.mahout.math.set.AbstractSet;
import org.junit.Assert;
import org.junit.Test;

public class OpenObject${valueTypeCap}HashMapTest extends Assert {

    private static class NotComparableKey {
    protected int x;
    
    public NotComparableKey(int x) {
      this.x = x;
    }
      
    @Override
    public String toString() {
      return "[k " + x + " ]";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + x;
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      NotComparableKey other = (NotComparableKey) obj;
      if (x != other.x) return false;
      return true;
    }
  }
  
   private static class ComparableKey extends NotComparableKey
      implements Comparable<ComparableKey> {
    public ComparableKey(int x) {
      super(x);
    }
      
    @Override
    public String toString() {
      return "[ck " + x + " ]";
    }

    @Override
    public int compareTo(ComparableKey o) {

      return (int)(x - o.x);
    }
  }
  
  private NotComparableKey[] ncKeys = {
    new NotComparableKey(101),
    new NotComparableKey(99),
    new NotComparableKey(2),
    new NotComparableKey(3),
    new NotComparableKey(4),
    new NotComparableKey(5)
    };
  
  private ComparableKey[] cKeys = {
    new ComparableKey(101),
    new ComparableKey(99),
    new ComparableKey(2),
    new ComparableKey(3),
    new ComparableKey(4),
    new ComparableKey(5)
    };
  

  @Test
  public void testConstructors() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.defaultCapacity, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new OpenObject${valueTypeCap}HashMap<ComparableKey>(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    
    map = new OpenObject${valueTypeCap}HashMap<ComparableKey>(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    int prime = PrimeFinder.nextPrime(907);
    
    map.ensureCapacity(prime);
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
  }
  
  @Test
  public void testClear() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType})11); 
    assertEquals(1, map.size());
    map.clear();
    assertEquals(0, map.size());
  }
  
  @Test
  @SuppressWarnings("unchecked")
  public void testClone() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType})11);
    OpenObject${valueTypeCap}HashMap<ComparableKey> map2 = (OpenObject${valueTypeCap}HashMap<ComparableKey>) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContainsKey() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType})11);
    assertTrue(map.containsKey(cKeys[0]));
    assertFalse(map.containsKey(cKeys[1]));
  }
  
  @Test
  public void testContainValue() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType})11);
    assertTrue(map.containsValue((${valueType})11));
    assertFalse(map.containsValue((${valueType})12));
  }
  
  @Test
  public void testForEachKey() {
    final List<ComparableKey> keys = new ArrayList<ComparableKey>();
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    map.forEachKey(new ObjectProcedure<ComparableKey>() {
      
      @Override
      public boolean apply(ComparableKey element) {
        keys.add(element);
        return true;
      }
    });
    
    //2, 3, 1, 0
    assertEquals(3, keys.size());
    Collections.sort(keys);
    assertSame(cKeys[3], keys.get(0));
    assertSame(cKeys[1], keys.get(1));
    assertSame(cKeys[0], keys.get(2));
  }
  
  private static class Pair implements Comparable<Pair> {
    ${valueType} v;
    ComparableKey k;
    
    Pair(ComparableKey k, ${valueType} v) {
      this.k = k;
      this.v = v;
    }
    
    @Override
    public int compareTo(Pair o) {
      return k.compareTo(o.k);
    }
  }
  
  @Test
  public void testForEachPair() {
    final List<Pair> pairs = new ArrayList<Pair>();
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    map.forEachPair(new Object${valueTypeCap}Procedure<ComparableKey>() {
      
      @Override
      public boolean apply(ComparableKey first, ${valueType} second) {
        pairs.add(new Pair(first, second));
        return true;
      }
    });
    
    Collections.sort(pairs);
    assertEquals(3, pairs.size());
    assertEquals((${valueType})14, pairs.get(0).v ${valueEpsilon});
    assertSame(cKeys[3], pairs.get(0).k);
    assertEquals((${valueType}) 12, pairs.get(1).v ${valueEpsilon});
    assertSame(cKeys[1], pairs.get(1).k);
    assertEquals((${valueType}) 11, pairs.get(2).v ${valueEpsilon});
    assertSame(cKeys[0], pairs.get(2).k);
    
    pairs.clear();
    map.forEachPair(new Object${valueTypeCap}Procedure<ComparableKey>() {
      int count = 0;
      
      @Override
      public boolean apply(ComparableKey first, ${valueType} second) {
        pairs.add(new Pair(first, second));
        count++;
        return count < 2;
      }
    });
    
    assertEquals(2, pairs.size());
  }
  
  @Test
  public void testGet() {
    OpenObject${valueTypeCap}HashMap<NotComparableKey> map = new OpenObject${valueTypeCap}HashMap<NotComparableKey>();
    map.put(ncKeys[0], (${valueType}) 11);
    map.put(ncKeys[1], (${valueType}) 12);
    assertEquals((${valueType})11, map.get(ncKeys[0]) ${valueEpsilon});
  }

  @Test
  public void testKeys() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    List<ComparableKey> keys = new ArrayList<ComparableKey>();
    map.keys(keys);
    Collections.sort(keys);
    assertSame(cKeys[1], keys.get(0));
    assertSame(cKeys[0], keys.get(1));
    List<ComparableKey> k2 = map.keys();
    Collections.sort(k2);
    assertEquals(keys, k2);
  }
  
  @Test
  public void testPairsMatching() {
    List<ComparableKey> keyList = new ArrayList<ComparableKey>();
    ${valueTypeCap}ArrayList valueList = new ${valueTypeCap}ArrayList();
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    map.pairsMatching(new Object${valueTypeCap}Procedure<ComparableKey>() {

      @Override
      public boolean apply(ComparableKey first, ${valueType} second) {
        return (second % 2) == 0;
      }},
        keyList, valueList);
    Collections.sort(keyList);
    valueList.sort();
    assertEquals(2, keyList.size());
    assertEquals(2, valueList.size());
    assertSame(cKeys[3], keyList.get(0));
    assertSame(cKeys[1], keyList.get(1));
    assertEquals((${valueType})14, valueList.get(1) ${valueEpsilon});
    assertEquals((${valueType})12, valueList.get(0) ${valueEpsilon});
  }
  
  @Test
  public void testValues() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    ${valueTypeCap}ArrayList values = new ${valueTypeCap}ArrayList(100);
    map.values(values);
    assertEquals(3, values.size());
    values.sort();
    assertEquals(11, values.get(0) ${valueEpsilon});
    assertEquals(12, values.get(1) ${valueEpsilon});
    assertEquals(14, values.get(2) ${valueEpsilon});
  }
  
  // tests of the code in the abstract class
  
  @Test
  public void testCopy() {
    OpenObject${valueTypeCap}HashMap<NotComparableKey> map = new OpenObject${valueTypeCap}HashMap<NotComparableKey>();
    map.put(ncKeys[0], (${valueType})11);
    OpenObject${valueTypeCap}HashMap<NotComparableKey> map2 = (OpenObject${valueTypeCap}HashMap<NotComparableKey>) map.copy();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testEquals() {
    // since there are no other subclasses of 
    // Abstractxxx available, we have to just test the
    // obvious.
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    OpenObject${valueTypeCap}HashMap<ComparableKey> map2 = (OpenObject${valueTypeCap}HashMap<ComparableKey>) map.copy();
    assertTrue(map.equals(map2));
    assertTrue(map2.equals(map));
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.removeKey(cKeys[0]);
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
  }
  
  // keys() tested in testKeys
  
  @Test
  public void testKeysSortedByValue() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    map.removeKey(cKeys[2]);
    List<ComparableKey> keys = new ArrayList<ComparableKey>();
    map.keysSortedByValue(keys);
    ComparableKey[] keysArray = keys.toArray(new ComparableKey[keys.size()]);
    assertArrayEquals(new ComparableKey[] {cKeys[0], cKeys[1], cKeys[3]},
        keysArray);
  }
  
  @Test
  public void testPairsSortedByKey() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    
    ${valueTypeCap}ArrayList values = new ${valueTypeCap}ArrayList();
    List<ComparableKey> keys = new ArrayList<ComparableKey>();
    map.pairsSortedByKey(keys, values);
    
    assertEquals(4, keys.size());
    assertEquals(4, values.size());
    assertEquals((${valueType}) 13, values.get(0) ${valueEpsilon});
    assertSame(cKeys[2], keys.get(0));
    assertEquals((${valueType}) 14, values.get(1) ${valueEpsilon});
    assertSame(cKeys[3], keys.get(1));
    assertEquals((${valueType}) 12, values.get(2) ${valueEpsilon});
    assertSame(cKeys[1], keys.get(2));
    assertEquals((${valueType}) 11, values.get(3) ${valueEpsilon});
    assertSame(cKeys[0], keys.get(3));
  }
  
  @Test
  public void testPairsSortedByValue() {
    OpenObject${valueTypeCap}HashMap<ComparableKey> map = new OpenObject${valueTypeCap}HashMap<ComparableKey>();
    map.put(cKeys[0], (${valueType}) 11);
    map.put(cKeys[1], (${valueType}) 12);
    map.put(cKeys[2], (${valueType}) 13);
    map.put(cKeys[3], (${valueType}) 14);
    
    List<ComparableKey> keys = new ArrayList<ComparableKey>();
    ${valueTypeCap}ArrayList values = new ${valueTypeCap}ArrayList();
    map.pairsSortedByValue(keys, values);
    assertEquals((${valueType}) 11, values.get(0) ${valueEpsilon});
    assertEquals(cKeys[0], keys.get(0));
    assertEquals((${valueType}) 12, values.get(1) ${valueEpsilon});
    assertEquals(cKeys[1], keys.get(1));
    assertEquals((${valueType}) 13, values.get(2) ${valueEpsilon});
    assertEquals(cKeys[2], keys.get(2));
    assertEquals((${valueType}) 14, values.get(3) ${valueEpsilon});
    assertEquals(cKeys[3], keys.get(3));
  }
 
 }
