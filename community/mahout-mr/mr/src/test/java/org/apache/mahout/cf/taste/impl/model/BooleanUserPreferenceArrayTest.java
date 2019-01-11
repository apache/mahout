/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.junit.Test;

public final class BooleanUserPreferenceArrayTest extends TasteTestCase {

  @Test
  public void testUserID() {
    PreferenceArray prefs = new BooleanUserPreferenceArray(3);
    assertEquals(3, prefs.length());
    prefs.setUserID(0, 1L);
    assertEquals(1L, prefs.getUserID(0));
    assertEquals(1L, prefs.getUserID(1));
    assertEquals(1L, prefs.getUserID(2));
  }

  @Test
  public void testItemID() {
    PreferenceArray prefs = new BooleanUserPreferenceArray(3);
    assertEquals(3, prefs.length());
    prefs.setItemID(0, 1L);
    prefs.setItemID(1, 2L);
    prefs.setItemID(2, 3L);
    assertEquals(1L, prefs.getItemID(0));
    assertEquals(2L, prefs.getItemID(1));
    assertEquals(3L, prefs.getItemID(2));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void testSetValue() {
    PreferenceArray prefs = new BooleanUserPreferenceArray(3);
    assertEquals(1.0f, prefs.getValue(2), EPSILON);
    assertEquals(3, prefs.length());
    prefs.setValue(0, 1.0f);
  }

  @Test
  public void testHasPref() {
    PreferenceArray prefs = new BooleanUserPreferenceArray(3);
    prefs.set(0, new GenericPreference(1L, 3L, 5.0f));
    assertTrue(prefs.hasPrefWithItemID(3L));
    assertTrue(prefs.hasPrefWithUserID(1L));
    assertFalse(prefs.hasPrefWithItemID(2L));
    assertFalse(prefs.hasPrefWithUserID(2L));
  }

  @Test
  public void testSort() {
    PreferenceArray prefs = new BooleanUserPreferenceArray(3);
    prefs.set(0, new BooleanPreference(1L, 3L));
    prefs.set(1, new BooleanPreference(1L, 1L));
    prefs.set(2, new BooleanPreference(1L, 2L));
    prefs.sortByItem();
    assertEquals(1L, prefs.getItemID(0));
    assertEquals(2L, prefs.getItemID(1));
    assertEquals(3L, prefs.getItemID(2));
  }

  @Test  
  public void testClone() {
    BooleanUserPreferenceArray prefs = new BooleanUserPreferenceArray(3);
    prefs.set(0, new BooleanPreference(1L, 3L));
    prefs.set(1, new BooleanPreference(1L, 1L));
    prefs.set(2, new BooleanPreference(1L, 2L));
    prefs = prefs.clone();
    assertEquals(3L, prefs.getItemID(0));
    assertEquals(1L, prefs.getUserID(1));
  }

}