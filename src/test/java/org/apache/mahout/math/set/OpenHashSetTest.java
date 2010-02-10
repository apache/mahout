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
 
  
package org.apache.mahout.math.set;
 
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.math.function.ObjectProcedure;
import org.apache.mahout.math.map.PrimeFinder;

import org.junit.Assert;
import org.junit.Test;

public class OpenHashSetTest extends Assert {
  
  @Test
  public void testConstructors() {
    OpenCharHashSet map = new OpenCharHashSet();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.defaultCapacity, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new OpenCharHashSet(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.defaultMaxLoadFactor, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.defaultMinLoadFactor, minLoadFactor[0], 0.001);
    
    map = new OpenCharHashSet(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    OpenCharHashSet map = new OpenCharHashSet();
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
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add("Horsefeathers");
    assertEquals(1, map.size());
    map.clear();
    assertEquals(0, map.size());
  }
  
  @SuppressWarnings("unchecked")
  @Test
  public void testClone() {
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add((char) 11);
    OpenHashSet<String> map2 = (OpenHashSet<String>) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContains() {
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add("Horsefeathers");
    assertTrue(map.contains(new String("Horsefeathers")));
    assertFalse(map.contains("Cowfeathers"));
  }
  
  @Test
  public void testForEachKey() {
    final List<String> keys = new ArrayList<String>();
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add("Eleven");
    map.add("Twelve");
    map.add("Thirteen");
    map.add("Fourteen");
    map.remove(new String("Thirteen"));
    map.forEachKey(new ObjectProcedure<String>() {
      
      @Override
      public boolean apply(String element) {
        keys.add(element);
        return true;
      }
    });
    
    String[] keysArray = keys.toArray(new String[keys.size()]);
    Arrays.sort(keysArray);
    
    assertArrayEquals(new String[] { new String("Eleven"), 
        new String("Fourteen"),
        new String("Twelve")}, 
        keysArray );
  }
  
  @Test
  public void testKeys() {
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add("Eleven");
    map.add("Twelve");
    List<String> keys = new ArrayList<String>();
    map.keys(keys);
    Collections.sort(keys);
    assertEquals(new String("Eleven"), keys.get(0));
    assertEquals(new String("Twelve"), keys.get(1));
    List<String> k2 = map.keys();
    Collections.sort(k2);
    assertEquals(keys, k2);
  }
  
  @SuppressWarnings("unchecked")
  @Test
  public void testEquals() {
    OpenHashSet<String> map = new OpenHashSet<String>();
    map.add("11");
    map.add("12");
    map.add("13");
    map.add("14");
    map.remove("13");
    OpenHashSet<String> map2 = (OpenHashSet<String>) map.clone();
    assertEquals(map, map2);
    assertEquals(map2, map);
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.remove("11");
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
  }
 }
