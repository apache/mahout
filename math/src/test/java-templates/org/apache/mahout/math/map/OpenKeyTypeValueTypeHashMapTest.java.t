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
 
#if (${keyTypeFloating} == 'true')
#set ($keyEpsilon = ", (${keyType})0.000001")
#else 
#set ($keyEpsilon = "")
#end
#if (${valueTypeFloating} == 'true')
#set ($valueEpsilon = ", (${valueType})0.000001")
#else 
#set ($valueEpsilon = "")
#end
  
 package org.apache.mahout.math.map;
 
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.math.function.${keyTypeCap}${valueTypeCap}Procedure;
import org.apache.mahout.math.function.${keyTypeCap}Procedure;
import org.apache.mahout.math.list.${keyTypeCap}ArrayList;
#if (${keyType} != ${valueType})
import org.apache.mahout.math.list.${valueTypeCap}ArrayList;
#end
import org.apache.mahout.math.set.AbstractSet;

import org.junit.Assert;
import org.junit.Test;

public class Open${keyTypeCap}${valueTypeCap}HashMapTest extends Assert {

  
  @Test
  public void testConstructors() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.DEFAULT_CAPACITY, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new Open${keyTypeCap}${valueTypeCap}HashMap(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    
    map = new Open${keyTypeCap}${valueTypeCap}HashMap(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
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
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put((${keyType}) 11, (${valueType}) 22);
    assertEquals(1, map.size());
    map.clear();
    assertEquals(0, map.size());
    assertEquals(0, map.get((${keyType}) 11), 0.0000001);
  }
  
  @Test
  public void testClone() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put((${keyType}) 11, (${valueType}) 22);
    Open${keyTypeCap}${valueTypeCap}HashMap map2 = (Open${keyTypeCap}${valueTypeCap}HashMap) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContainsKey() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    assertTrue(map.containsKey(($keyType) 11));
    assertFalse(map.containsKey(($keyType) 12));
  }
  
  @Test
  public void testContainValue() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    assertTrue(map.containsValue((${valueType}) 22));
    assertFalse(map.containsValue((${valueType}) 23));
  }
  
  @Test
  public void testForEachKey() {
    final ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    map.forEachKey(new ${keyTypeCap}Procedure() {
      
      @Override
      public boolean apply(${keyType} element) {
        keys.add(element);
        return true;
      }
    });
    
    ${keyType}[] keysArray = keys.toArray(new ${keyType}[keys.size()]);
    Arrays.sort(keysArray);
    
    assertArrayEquals(new ${keyType}[] {11, 12, 14}, keysArray ${keyEpsilon});
  }
  
  private static class Pair implements Comparable<Pair> {
    ${keyType} k;
    ${valueType} v;
    
    Pair(${keyType} k, ${valueType} v) {
      this.k = k;
      this.v = v;
    }
    
    @Override
    public int compareTo(Pair o) {
      if (k < o.k) {
        return -1;
      } else if (k == o.k) {
        return 0;
      } else {
        return 1;
      }
    }
  }
  
  @Test
  public void testForEachPair() {
    final List<Pair> pairs = new ArrayList<Pair>();
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    map.forEachPair(new ${keyTypeCap}${valueTypeCap}Procedure() {
      
      @Override
      public boolean apply(${keyType} first, ${valueType} second) {
        pairs.add(new Pair(first, second));
        return true;
      }
    });
    
    Collections.sort(pairs);
    assertEquals(3, pairs.size());
    assertEquals(($keyType) 11, pairs.get(0).k ${keyEpsilon});
    assertEquals((${valueType}) 22, pairs.get(0).v ${valueEpsilon});
    assertEquals(($keyType) 12, pairs.get(1).k ${keyEpsilon});
    assertEquals((${valueType}) 23, pairs.get(1).v ${valueEpsilon});
    assertEquals(($keyType) 14, pairs.get(2).k ${keyEpsilon});
    assertEquals((${valueType}) 25, pairs.get(2).v ${valueEpsilon});
    
    pairs.clear();
    map.forEachPair(new ${keyTypeCap}${valueTypeCap}Procedure() {
      int count = 0;
      
      @Override
      public boolean apply(${keyType} first, ${valueType} second) {
        pairs.add(new Pair(first, second));
        count++;
        return count < 2;
      }
    });
    
    assertEquals(2, pairs.size());
  }
  
  @Test
  public void testGet() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    assertEquals(22, map.get(($keyType)11) ${valueEpsilon});
    assertEquals(0, map.get(($keyType)0) ${valueEpsilon});
  }
  
  @Test
  public void testAdjustOrPutValue() {
   Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.adjustOrPutValue((${keyType})11, (${valueType})1, (${valueType})3);
    assertEquals(25, map.get((${keyType})11) ${valueEpsilon});
    map.adjustOrPutValue((${keyType})15, (${valueType})1, (${valueType})3);
    assertEquals(1, map.get((${keyType})15) ${valueEpsilon});
  }
  
  @Test
  public void testKeys() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 22);
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    map.keys(keys);
    keys.sort();
    assertEquals(11, keys.get(0) ${keyEpsilon});
    assertEquals(12, keys.get(1) ${keyEpsilon});
    ${keyTypeCap}ArrayList k2 = map.keys();
    k2.sort();
    assertEquals(keys, k2);
  }
  
  @Test
  public void testPairsMatching() {
    ${keyTypeCap}ArrayList keyList = new ${keyTypeCap}ArrayList();
    ${valueTypeCap}ArrayList valueList = new ${valueTypeCap}ArrayList();
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    map.pairsMatching(new ${keyTypeCap}${valueTypeCap}Procedure() {

      @Override
      public boolean apply(${keyType} first, ${valueType} second) {
        return (first % 2) == 0;
      }},
        keyList, valueList);
    keyList.sort();
    valueList.sort();
    assertEquals(2, keyList.size());
    assertEquals(2, valueList.size());
    assertEquals(12, keyList.get(0) ${keyEpsilon});
    assertEquals(14, keyList.get(1) ${keyEpsilon});
    assertEquals(23, valueList.get(0) ${valueEpsilon});
    assertEquals(25, valueList.get(1) ${valueEpsilon});
  }
  
  @Test
  public void testValues() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    ${valueTypeCap}ArrayList values = new ${valueTypeCap}ArrayList(100);
    map.values(values);
    assertEquals(3, values.size());
    values.sort();
    assertEquals(22, values.get(0) ${valueEpsilon});
    assertEquals(23, values.get(1) ${valueEpsilon});
    assertEquals(25, values.get(2) ${valueEpsilon});
  }
  
  // tests of the code in the abstract class
  
  @Test
  public void testCopy() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    Open${keyTypeCap}${valueTypeCap}HashMap map2 = (Open${keyTypeCap}${valueTypeCap}HashMap) map.copy();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testEquals() {
    // since there are no other subclasses of 
    // Abstractxxx available, we have to just test the
    // obvious.
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    Open${keyTypeCap}${valueTypeCap}HashMap map2 = (Open${keyTypeCap}${valueTypeCap}HashMap) map.copy();
    assertEquals(map, map2);
    assertTrue(map2.equals(map));
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.removeKey(($keyType) 11);
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
  }
  
  // keys() tested in testKeys
  
  @Test
  public void testKeysSortedByValue() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 22);
    map.put(($keyType) 12, (${valueType}) 23);
    map.put(($keyType) 13, (${valueType}) 24);
    map.put(($keyType) 14, (${valueType}) 25);
    map.removeKey(($keyType) 13);
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    map.keysSortedByValue(keys);
    ${keyType}[] keysArray = keys.toArray(new ${keyType}[keys.size()]);
    assertArrayEquals(new ${keyType}[] {11, 12, 14},
        keysArray ${keyEpsilon});
  }
  
  @Test
  public void testPairsSortedByKey() {
    Open${keyTypeCap}${valueTypeCap}HashMap map = new Open${keyTypeCap}${valueTypeCap}HashMap();
    map.put(($keyType) 11, (${valueType}) 100);
    map.put(($keyType) 12, (${valueType}) 70);
    map.put(($keyType) 13, (${valueType}) 30);
    map.put(($keyType) 14, (${valueType}) 3);
    
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    ${valueTypeCap}ArrayList values = new ${valueTypeCap}ArrayList();
    map.pairsSortedByKey(keys, values);
    
    assertEquals(4, keys.size());
    assertEquals(4, values.size());
    assertEquals(($keyType) 11, keys.get(0) ${keyEpsilon});
    assertEquals((${valueType}) 100, values.get(0) ${valueEpsilon});
    assertEquals(($keyType) 12, keys.get(1) ${keyEpsilon});
    assertEquals((${valueType}) 70, values.get(1) ${valueEpsilon});
    assertEquals(($keyType) 13, keys.get(2) ${keyEpsilon});
    assertEquals((${valueType}) 30, values.get(2) ${valueEpsilon});
    assertEquals(($keyType) 14, keys.get(3) ${keyEpsilon});
    assertEquals((${valueType}) 3, values.get(3) ${valueEpsilon});
    keys.clear();
    values.clear();
    map.pairsSortedByValue(keys, values);
    assertEquals(($keyType) 11, keys.get(3) ${keyEpsilon});
    assertEquals((${valueType}) 100, values.get(3) ${valueEpsilon});
    assertEquals(($keyType) 12, keys.get(2) ${keyEpsilon});
    assertEquals((${valueType}) 70, values.get(2) ${valueEpsilon});
    assertEquals(($keyType) 13, keys.get(1) ${keyEpsilon});
    assertEquals((${valueType}) 30, values.get(1) ${valueEpsilon});
    assertEquals(($keyType) 14, keys.get(0) ${keyEpsilon});
    assertEquals(($valueType) 3, values.get(0) ${valueEpsilon});
  }
 
 }
