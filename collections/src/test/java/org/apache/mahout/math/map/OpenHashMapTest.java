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
  
package org.apache.mahout.math.map;
 
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.mahout.math.function.ObjectObjectProcedure;
import org.apache.mahout.math.function.ObjectProcedure;
import org.apache.mahout.math.set.AbstractSet;
import org.junit.Assert;
import org.junit.Test;

public class OpenHashMapTest extends Assert {

  @Test
  public void testConstructors() {
    OpenHashMap<String,String> map = new OpenHashMap<String, String>();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.defaultCapacity, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new OpenHashMap<String, String>(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    
    map = new OpenHashMap<String, String>(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
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
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11"); 
    assertEquals(1, map.size());
    map.clear();
    assertEquals(0, map.size());
  }
  
  @Test
  @SuppressWarnings("unchecked")
  public void testClone() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    OpenHashMap<String, String> map2 = (OpenHashMap<String, String>) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContainsKey() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    assertTrue(map.containsKey("Eleven"));
    assertFalse(map.containsKey("Twelve"));
  }
  
  @Test
  public void testContainValue() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    assertTrue(map.containsValue(new String("11")));
    assertFalse(map.containsValue("Cowfeathers"));
  }
  
  @Test
  public void testForEachKey() {
    final List<String> keys = new ArrayList<String>();
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    map.put("Thirteen", "13");
    map.put("Fourteen", "14");
    map.remove(new String("Thirteen"));
    map.forEachKey(new ObjectProcedure<String>() {
      
      @Override
      public boolean apply(String element) {
        keys.add(element);
        return true;
      }
    });
    
    //2, 3, 1, 0
    assertEquals(3, keys.size());
    Collections.sort(keys);
    assertSame("Eleven", keys.get(0));
    assertSame("Fourteen", keys.get(1));
    assertSame("Twelve", keys.get(2));
  }
  
  private static class Pair implements Comparable<Pair> {
    final String k;
    final String v;
    
    Pair(String k, String v) {
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
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    map.put("Thirteen", "13");
    map.put("Fourteen", "14");
    map.remove(new String("Thirteen"));
    map.forEachPair(new ObjectObjectProcedure<String, String>() {
      
      @Override
      public boolean apply(String first, String second) {
        pairs.add(new Pair(first, second));
        return true;
      }
    });
    
    Collections.sort(pairs);
    assertEquals(3, pairs.size());
    assertEquals("11", pairs.get(0).v );
    assertEquals("Eleven", pairs.get(0).k);
    assertEquals("14", pairs.get(1).v );
    assertEquals("Fourteen", pairs.get(1).k);
    assertEquals("12", pairs.get(2).v );
    assertSame("Twelve", pairs.get(2).k);
    
    pairs.clear();
    map.forEachPair(new ObjectObjectProcedure<String, String>() {
      int count = 0;
      
      @Override
      public boolean apply(String first, String second) {
        pairs.add(new Pair(first, second));
        count++;
        return count < 2;
      }
    });
    
    assertEquals(2, pairs.size());
  }
  
  @Test
  public void testGet() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    assertEquals("11", map.get(new String("Eleven")));
  }

  @Test
  public void testKeys() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    map.put("Thirteen", "13");
    map.put("Fourteen", "14");
    map.remove(new String("Thirteen"));
    List<String> keys = new ArrayList<String>();
    map.keys(keys);
    Collections.sort(keys);
    assertSame("Eleven", keys.get(0));
    assertSame("Fourteen", keys.get(1));
    Set<String> k2 = map.keySet();
    List<String> k2l = new ArrayList<String>();
    k2l.addAll(k2);
    Collections.sort(k2l);
    assertEquals(keys, k2l);
  }
  
  @Test
  public void testValues() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    map.put("Thirteen", "13");
    map.put("Fourteen", "14");
    map.remove(new String("Thirteen"));
    map.put("Extra", "11");
    Collection<String> values = map.values();
    assertEquals(4, values.size());
  }
  
  // tests of the code in the abstract class
  
  @SuppressWarnings("unchecked")
  @Test
  public void testCopy() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    OpenHashMap<String, String> map2 = (OpenHashMap<String, String>) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @SuppressWarnings("unchecked")
  @Test
  public void testEquals() {
    OpenHashMap<String, String> map = new OpenHashMap<String, String>();
    map.put("Eleven", "11");
    map.put("Twelve", "12");
    map.put("Thirteen", "13");
    map.put("Fourteen", "14");
    map.remove(new String("Thirteen"));
    OpenHashMap<String, String> map2 = (OpenHashMap<String, String>) map.clone();
    assertTrue(map.equals(map2));
    assertTrue(map2.equals(map));
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.remove("Eleven");
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
  }
 }
