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

package org.apache.mahout.utils;

import junit.framework.TestCase;

import java.util.Arrays;
import java.util.List;

public class StringUtilsTest extends TestCase {

  private static class DummyTest {
    public int field;

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null || !(obj instanceof DummyTest)) {
        return false;
      }

      DummyTest dt = (DummyTest) obj;
      return field == dt.field;
    }

    @Override
    public int hashCode() {
      return field;
    }

  }

  public void testStringConversion() throws Exception {

    List<String> expected = Arrays.asList("A", "B", "C");
    assertEquals(expected, StringUtils.fromString(StringUtils
        .toString(expected)));

    // test a non serializable object
    DummyTest test = new DummyTest();
    assertEquals(test, StringUtils.fromString(StringUtils.toString(test)));
  }

}
