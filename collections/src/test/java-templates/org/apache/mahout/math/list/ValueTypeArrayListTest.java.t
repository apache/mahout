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

package org.apache.mahout.math.list;

import org.apache.mahout.math.function.${valueTypeCap}Procedure;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

#if ($valueTypeFloating == 'true')
  #set ($assertEpsilon=', 0001')
#else
  #set ($assertEpsilon='')
#end

public class ${valueTypeCap}ArrayListTest extends Assert {
  
  private ${valueTypeCap}ArrayList emptyList;
  private ${valueTypeCap}ArrayList listOfFive;
  
  @Before
  public void before() {
    emptyList = new ${valueTypeCap}ArrayList();
    listOfFive = new ${valueTypeCap}ArrayList();
    for (int x = 0; x < 5; x ++) {
      listOfFive.add((${valueType})x);
    }
  }
  
  
  @Test(expected = IndexOutOfBoundsException.class)
  public void testGetEmpty() {
    emptyList.get(0);
  }
  
  @Test(expected = IndexOutOfBoundsException.class)
  public void setEmpty() {
    emptyList.set(1, (${valueType})1);
  }
  
  @Test(expected = IndexOutOfBoundsException.class)
  public void beforeInsertInvalidRange() {
    emptyList.beforeInsert(1, (${valueType})0);
  }
  
  @Test
  public void testAdd() {
    emptyList.add((${valueType})12);
    assertEquals(1, emptyList.size());
    for (int x = 0; x < 1000; x ++) {
      emptyList.add((${valueType})(x % ${valueObjectType}.MAX_VALUE));
    }
    assertEquals(1001, emptyList.size());
    assertEquals(12, emptyList.get(0) $assertEpsilon);
    for (int x = 0; x < 1000; x ++) {
      assertEquals((${valueType})(x % ${valueObjectType}.MAX_VALUE), emptyList.get(x+1) $assertEpsilon);
    }
  }
  
  @Test
  public void testBinarySearch() 
  {
    int x = listOfFive.binarySearchFromTo((${valueType})0, 2, 4);
    assertEquals(-3, x);
    x = listOfFive.binarySearchFromTo((${valueType})1, 0, 4);
    assertEquals(1, x);
  }
  
  @Test
  public void testClone() {
    ${valueTypeCap}ArrayList l2 = listOfFive.copy(); // copy just calls clone.
    assertNotSame(listOfFive, l2);
    assertEquals(listOfFive, l2);
  }
  
  @Test 
  public void testElements() {
    ${valueType}[] l = { 12, 24, 36, 48 };
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList(l);
    assertEquals(4, lar.size());
    assertSame(l, lar.elements());
    ${valueType}[] l2 = { 3, 6, 9, 12 };
    lar.elements(l2);
    assertSame(l2, lar.elements());
  }
  
  @Test
  public void testEquals() {
    ${valueType}[] l = { 12, 24, 36, 48 };
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList(l);
    ${valueTypeCap}ArrayList lar2 = new ${valueTypeCap}ArrayList();
    for (int x = 0; x < lar.size(); x++) {
      lar2.add(lar.get(x));
    }
    assertEquals(lar, lar2);
    assertFalse(lar.equals(this));
    lar2.add((${valueType})55);
    assertFalse(lar.equals(lar2));
  }

  @Test
  public void testForEach() {
    listOfFive.forEach(new ${valueTypeCap}Procedure() {
      int count;
      @Override
      public boolean apply(${valueType} element) {
        assertFalse(count > 2);
        count ++;
        return element != 1;
      }});
  }
  
  @Test
  public void testGetQuick() {
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList(10);
    lar.getQuick(1); // inside capacity, outside size.
  }
  
  @Test
  public void testIndexOfFromTo() {
    int x = listOfFive.indexOfFromTo((${valueType})0, 2, 4);
    assertEquals(-1, x);
    x = listOfFive.indexOfFromTo((${valueType})1, 0, 4);
    assertEquals(1, x);
  }
  
  @Test
  public void testLastIndexOfFromTo() {
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList(10);
    lar.add((${valueType})1);
    lar.add((${valueType})2);
    lar.add((${valueType})3);
    lar.add((${valueType})2);
    lar.add((${valueType})1);
    assertEquals(3, lar.lastIndexOf((${valueType})2));
    assertEquals(3, lar.lastIndexOfFromTo((${valueType})2, 2, 4));
    assertEquals(-1, lar.lastIndexOf((${valueType})111));
  }
  
  @Test
  public void testPartFromTo() {
    Abstract${valueTypeCap}List al = listOfFive.partFromTo(1, 2);
    assertEquals(2, al.size());
    assertEquals(1, al.get(0) $assertEpsilon);
    assertEquals(2, al.get(1) $assertEpsilon);
  }
  
  @Test(expected = IndexOutOfBoundsException.class)
  public void testPartFromToOOB() {
    listOfFive.partFromTo(10, 11);
  }
  
  @Test
  public void testRemoveAll() {
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList(1000);
    for (int x = 0; x < 128; x ++) {
      lar.add((${valueType})x);
    }
    ${valueTypeCap}ArrayList larOdd = new ${valueTypeCap}ArrayList(500);
    for (int x = 1; x < 128; x = x + 2) {
      larOdd.add((${valueType})x);
    }
    lar.removeAll(larOdd);
    assertEquals(64, lar.size());
    
    for (int x = 0; x < lar.size(); x++) {
      assertEquals(x*2, lar.get(x) $assertEpsilon);
    }
  }
  
  @Test
  public void testReplaceFromToWith() {
    listOfFive.add((${valueType})5);
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList();
    lar.add((${valueType})44);
    lar.add((${valueType})55);
    listOfFive.replaceFromToWithFromTo(2, 3, lar, 0, 1);
    assertEquals(0, listOfFive.get(0) $assertEpsilon);
    assertEquals(1, listOfFive.get(1) $assertEpsilon);
    assertEquals(44, listOfFive.get(2) $assertEpsilon);
    assertEquals(55, listOfFive.get(3) $assertEpsilon);
    assertEquals(4, listOfFive.get(4) $assertEpsilon);
    assertEquals(5, listOfFive.get(5) $assertEpsilon);
  }
  
  @Test
  public void testRetainAllSmall() {
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList();
    lar.addAllOf(listOfFive);
    lar.addAllOf(listOfFive);
    lar.addAllOf(listOfFive);
    ${valueTypeCap}ArrayList lar2 = new ${valueTypeCap}ArrayList();
    lar2.add((${valueType})3);
    lar2.add((${valueType})4);
    assertTrue(lar.retainAll(lar2));
    for(int x = 0; x < lar.size(); x ++) {
      ${valueType} l = lar.get(x);
      assertTrue(l == 3 || l == 4);
    }
    assertEquals(6, lar.size());
  }
  
  @Test
  public void testRetainAllSmaller() {
    ${valueTypeCap}ArrayList lar = new ${valueTypeCap}ArrayList();
    lar.addAllOf(listOfFive);
    ${valueTypeCap}ArrayList lar2 = new ${valueTypeCap}ArrayList();
    // large 'other' arg to take the other code path.
    for (int x = 0; x < 1000; x ++) {
      lar2.add((${valueType})3);
      lar2.add((${valueType})4);
    }
    assertTrue(lar.retainAll(lar2));
    for(int x = 0; x < lar.size(); x ++) {
      ${valueType} l = lar.get(x);
      assertTrue(l == 3 || l == 4);
    }
  }

}
