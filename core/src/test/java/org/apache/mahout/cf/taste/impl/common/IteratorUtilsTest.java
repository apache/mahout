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

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.TasteTestCase;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.TreeSet;

public final class IteratorUtilsTest extends TasteTestCase {

  private static final List<String> TEST_DATA;

  static {
    List<String> temp = new ArrayList<String>(3);
    temp.add("bar");
    temp.add("baz");
    temp.add("foo");
    TEST_DATA = Collections.unmodifiableList(temp);
  }

  public void testArray() {
    String[] data = TEST_DATA.toArray(new String[3]);
    assertEquals(TEST_DATA, IteratorUtils.iterableToList(new ArrayIterator<String>(data)));
  }

  public void testList() {
    assertEquals(TEST_DATA, IteratorUtils.iterableToList(TEST_DATA));
  }

  public void testCollection() {
    Collection<String> data = new TreeSet<String>();
    data.add("foo");
    data.add("bar");
    data.add("baz");
    assertEquals(TEST_DATA, IteratorUtils.iterableToList(data));
  }

  public void testComparator() {
    Collection<String> data = new ArrayList<String>(3);
    data.add("baz");
    data.add("bar");
    data.add("foo");
    assertEquals(TEST_DATA, IteratorUtils.iterableToList(data, String.CASE_INSENSITIVE_ORDER));
  }

  public void testEmpty() {
    assertEquals(0, IteratorUtils.iterableToList(new ArrayList<Object>(0)).size());
  }

}
