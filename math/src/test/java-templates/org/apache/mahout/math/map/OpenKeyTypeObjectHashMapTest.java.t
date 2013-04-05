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
  
 package org.apache.mahout.math.map;
 
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.math.function.${keyTypeCap}ObjectProcedure;
import org.apache.mahout.math.function.${keyTypeCap}Procedure;
import org.apache.mahout.math.list.${keyTypeCap}ArrayList;
import org.apache.mahout.math.set.AbstractSet;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class Open${keyTypeCap}ObjectHashMapTest extends Assert {
  
  private static class TestClass implements Comparable<TestClass>{
    
    TestClass(${keyType} x) {
      this.x = x;
    }
    
    @Override
    public String toString() {
      return "[ts " + x + " ]";
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ${keyObjectType}.valueOf(x).hashCode();
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      TestClass other = (TestClass) obj;
      return x == other.x;
    }

    ${keyType} x;

    @Override
    public int compareTo(TestClass o) {

#if (${keyTypeFloating} == 'true')
      return ${keyObjectType}.compare(x, o.x);
#else
      return (int)(x - o.x);
#end
    }
  }
  
  private TestClass item;
  private TestClass anotherItem;
  private TestClass anotherItem2;
  private TestClass anotherItem3;
  private TestClass anotherItem4;
  private TestClass anotherItem5;
  
  @Before
  public void before() {
    item = new TestClass((${keyType})101);
    anotherItem = new TestClass((${keyType})99);
    anotherItem2 = new TestClass((${keyType})2);
    anotherItem3 = new TestClass((${keyType})3);
    anotherItem4 = new TestClass((${keyType})4);
    anotherItem5 = new TestClass((${keyType})5);
    
  }

  
  @Test
  public void testConstructors() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.DEFAULT_CAPACITY, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new Open${keyTypeCap}ObjectHashMap<TestClass>(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    
    map = new Open${keyTypeCap}ObjectHashMap<TestClass>(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
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
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    for (int i = 0; i < 100; i++) {
      map.put((${keyType}) i, item);
      assertEquals(1, map.size());
      map.clear();
      assertEquals(0, map.size());
      assertFalse("Contains: " + i, map.containsKey((${keyType}) i));
      assertSame(null, map.get((${keyType}) i));
    }
  }

  @Test
  public void testClone() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    Open${keyTypeCap}ObjectHashMap<TestClass> map2 = (Open${keyTypeCap}ObjectHashMap<TestClass>) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContainsKey() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    assertTrue(map.containsKey((${keyType}) 11));
    assertFalse(map.containsKey((${keyType}) 12));
  }
  
  @Test
  public void testContainValue() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    assertTrue(map.containsValue(item));
    assertFalse(map.containsValue(anotherItem));
  }
  
  @Test
  public void testForEachKey() {
    final ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem);
    map.put((${keyType}) 12, anotherItem2);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem4);
    map.removeKey((${keyType}) 13);
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
    TestClass v;
    
    Pair(${keyType} k, TestClass v) {
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
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem);
    map.put((${keyType}) 12, anotherItem2);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem4);
    map.removeKey((${keyType}) 13);
    map.forEachPair(new ${keyTypeCap}ObjectProcedure<TestClass>() {
      
      @Override
      public boolean apply(${keyType} first, TestClass second) {
        pairs.add(new Pair(first, second));
        return true;
      }
    });
    
    Collections.sort(pairs);
    assertEquals(3, pairs.size());
    assertEquals((${keyType}) 11, pairs.get(0).k ${keyEpsilon});
    assertSame(anotherItem, pairs.get(0).v );
    assertEquals((${keyType}) 12, pairs.get(1).k ${keyEpsilon});
    assertSame(anotherItem2, pairs.get(1).v );
    assertEquals((${keyType}) 14, pairs.get(2).k ${keyEpsilon});
    assertSame(anotherItem4, pairs.get(2).v );
    
    pairs.clear();
    map.forEachPair(new ${keyTypeCap}ObjectProcedure<TestClass>() {
      int count = 0;
      
      @Override
      public boolean apply(${keyType} first, TestClass second) {
        pairs.add(new Pair(first, second));
        count++;
        return count < 2;
      }
    });
    
    assertEquals(2, pairs.size());
  }
  
  @Test
  public void testGet() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    map.put((${keyType}) 12, anotherItem);
    assertSame(item, map.get((${keyType})11) );
    assertSame(null, map.get((${keyType})0) );
  }

  @Test
  public void testKeys() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    map.put((${keyType}) 12, item);
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
    List<TestClass> valueList = new ArrayList<TestClass>();
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem2);
    map.put((${keyType}) 12, anotherItem3);
    map.put((${keyType}) 13, anotherItem4);
    map.put((${keyType}) 14, anotherItem5);
    map.removeKey((${keyType}) 13);
    map.pairsMatching(new ${keyTypeCap}ObjectProcedure<TestClass>() {

      @Override
      public boolean apply(${keyType} first, TestClass second) {
        return (first % 2) == 0;
      }},
        keyList, valueList);
    keyList.sort();
    Collections.sort(valueList);
    assertEquals(2, keyList.size());
    assertEquals(2, valueList.size());
    assertEquals(12, keyList.get(0) ${keyEpsilon});
    assertEquals(14, keyList.get(1) ${keyEpsilon});
    assertSame(anotherItem3, valueList.get(0) );
    assertSame(anotherItem5, valueList.get(1) );
  }
  
  @Test
  public void testValues() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem);
    map.put((${keyType}) 12, anotherItem2);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem4);
    map.removeKey((${keyType}) 13);
    List<TestClass> values = new ArrayList<TestClass>(100);
    map.values(values);
    assertEquals(3, values.size());
    Collections.sort(values);
    assertEquals(anotherItem2, values.get(0) );
    assertEquals(anotherItem4, values.get(1) );
    assertEquals(anotherItem, values.get(2) );
  }
  
  // tests of the code in the abstract class
  
  @Test
  public void testCopy() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, item);
    Open${keyTypeCap}ObjectHashMap<TestClass> map2 = (Open${keyTypeCap}ObjectHashMap<TestClass>) map.copy();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testEquals() {
    // since there are no other subclasses of 
    // Abstractxxx available, we have to just test the
    // obvious.
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem);
    map.put((${keyType}) 12, anotherItem2);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem4);
    map.removeKey((${keyType}) 13);
    Open${keyTypeCap}ObjectHashMap<TestClass> map2 = (Open${keyTypeCap}ObjectHashMap<TestClass>) map.copy();
    assertEquals(map, map2);
    assertTrue(map2.equals(map));
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.removeKey((${keyType}) 11);
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
  }
  
  // keys() tested in testKeys
  
  @Test
  public void testKeysSortedByValue() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem5);
    map.put((${keyType}) 12, anotherItem4);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem2);
    map.removeKey((${keyType}) 13);
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    map.keysSortedByValue(keys);
    ${keyType}[] keysArray = keys.toArray(new ${keyType}[keys.size()]);
    assertArrayEquals(new ${keyType}[] {14, 12, 11},
        keysArray ${keyEpsilon});
  }
  
  @Test
  public void testPairsSortedByKey() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem5);
    map.put((${keyType}) 12, anotherItem4);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem2);
    
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    List<TestClass> values = new ArrayList<TestClass>();
    map.pairsSortedByKey(keys, values);
    
    assertEquals(4, keys.size());
    assertEquals(4, values.size());
    assertEquals((${keyType}) 11, keys.get(0) ${keyEpsilon});
    assertSame(anotherItem5, values.get(0) );
    assertEquals((${keyType}) 12, keys.get(1) ${keyEpsilon});
    assertSame(anotherItem4, values.get(1) );
    assertEquals((${keyType}) 13, keys.get(2) ${keyEpsilon});
    assertSame(anotherItem3, values.get(2) );
    assertEquals((${keyType}) 14, keys.get(3) ${keyEpsilon});
    assertSame(anotherItem2, values.get(3) );
  }
  
  @Test
  public void testPairsSortedByValue() {
    Open${keyTypeCap}ObjectHashMap<TestClass> map = new Open${keyTypeCap}ObjectHashMap<TestClass>();
    map.put((${keyType}) 11, anotherItem5);
    map.put((${keyType}) 12, anotherItem4);
    map.put((${keyType}) 13, anotherItem3);
    map.put((${keyType}) 14, anotherItem2);
    
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    List<TestClass> values = new ArrayList<TestClass>();
    map.pairsSortedByValue(keys, values);
    assertEquals((${keyType}) 11, keys.get(3) ${keyEpsilon});
    assertEquals(anotherItem5, values.get(3) );
    assertEquals((${keyType}) 12, keys.get(2) ${keyEpsilon});
    assertEquals(anotherItem4, values.get(2) );
    assertEquals((${keyType}) 13, keys.get(1) ${keyEpsilon});
    assertEquals(anotherItem3, values.get(1) );
    assertEquals((${keyType}) 14, keys.get(0) ${keyEpsilon});
    assertEquals(anotherItem2, values.get(0) );
  }
 
 }
