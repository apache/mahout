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
  
package org.apache.mahout.math.set;
 
import java.util.Arrays;

import org.apache.mahout.math.function.${keyTypeCap}Procedure;
import org.apache.mahout.math.list.${keyTypeCap}ArrayList;
import org.apache.mahout.math.map.PrimeFinder;

import org.junit.Assert;
import org.junit.Test;

public class Open${keyTypeCap}HashSetTest extends Assert {

  
  @Test
  public void testConstructors() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    int[] capacity = new int[1];
    double[] minLoadFactor = new double[1];
    double[] maxLoadFactor = new double[1];
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(AbstractSet.DEFAULT_CAPACITY, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    int prime = PrimeFinder.nextPrime(907);
    map = new Open${keyTypeCap}HashSet(prime);
    
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(AbstractSet.DEFAULT_MAX_LOAD_FACTOR, maxLoadFactor[0], 0.001);
    assertEquals(AbstractSet.DEFAULT_MIN_LOAD_FACTOR, minLoadFactor[0], 0.001);
    
    map = new Open${keyTypeCap}HashSet(prime, 0.4, 0.8);
    map.getInternalFactors(capacity, minLoadFactor, maxLoadFactor);
    assertEquals(prime, capacity[0]);
    assertEquals(0.4, minLoadFactor[0], 0.001);
    assertEquals(0.8, maxLoadFactor[0], 0.001);
  }
  
  @Test
  public void testEnsureCapacity() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
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
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add((${keyType}) 11);
    assertEquals(1, map.size());
    map.clear();
    assertEquals(0, map.size());
  }
  
  @Test
  public void testClone() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add((${keyType}) 11);
    Open${keyTypeCap}HashSet map2 = (Open${keyTypeCap}HashSet) map.clone();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testContains() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add(($keyType) 11);
    assertTrue(map.contains(($keyType) 11));
    assertFalse(map.contains(($keyType) 12));
  }
  
  @Test
  public void testForEachKey() {
    final ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add(($keyType) 11);
    map.add(($keyType) 12);
    map.add(($keyType) 13);
    map.add(($keyType) 14);
    map.remove(($keyType) 13);
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
  
  @Test
  public void testKeys() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add(($keyType) 11);
    map.add(($keyType) 12);
    ${keyTypeCap}ArrayList keys = new ${keyTypeCap}ArrayList();
    map.keys(keys);
    keys.sort();
    assertEquals(11, keys.get(0) ${keyEpsilon});
    assertEquals(12, keys.get(1) ${keyEpsilon});
    ${keyTypeCap}ArrayList k2 = map.keys();
    k2.sort();
    assertEquals(keys, k2);
  }
  
  // tests of the code in the abstract class
  
  @Test
  public void testCopy() {
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add(($keyType) 11);
    Open${keyTypeCap}HashSet map2 = (Open${keyTypeCap}HashSet) map.copy();
    map.clear();
    assertEquals(1, map2.size());
  }
  
  @Test
  public void testEquals() {
    // since there are no other subclasses of 
    // Abstractxxx available, we have to just test the
    // obvious.
    Open${keyTypeCap}HashSet map = new Open${keyTypeCap}HashSet();
    map.add(($keyType) 11);
    map.add(($keyType) 12);
    map.add(($keyType) 13);
    map.add(($keyType) 14);
    map.remove(($keyType) 13);
    Open${keyTypeCap}HashSet map2 = (Open${keyTypeCap}HashSet) map.copy();
    assertTrue(map.equals(map2));
    assertTrue(map.hashCode() == map2.hashCode());
    assertTrue(map2.equals(map));
    assertTrue(map.hashCode() == map2.hashCode());
    assertFalse("Hello Sailor".equals(map));
    assertFalse(map.equals("hello sailor"));
    map2.remove(($keyType) 11);
    assertFalse(map.equals(map2));
    assertFalse(map2.equals(map));
    assertFalse(map.hashCode() == map2.hashCode());
  }
 }
