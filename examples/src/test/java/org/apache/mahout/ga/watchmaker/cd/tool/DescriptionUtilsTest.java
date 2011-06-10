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

package org.apache.mahout.ga.watchmaker.cd.tool;

import com.google.common.collect.Lists;
import org.apache.mahout.examples.MahoutTestCase;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public final class DescriptionUtilsTest extends MahoutTestCase {

  @Test
  public void testCreateNominalDescription() {
    List<String> values = Arrays.asList("val1", "val2", "val3");

    String description = DescriptionUtils.createNominalDescription(values);

    assertEquals("val1,val2,val3", description);
  }

  @Test
  public void testCreateNumericalDescription() {
    String description = DescriptionUtils.createNumericalDescription(-5.1, 12.32);
    assertEquals("-5.1,12.32", description);
  }

  @Test
  public void testExtractNominalValues() {
    String description = "val1,val2,val3";
    Collection<String> target = Lists.newArrayList();

    DescriptionUtils.extractNominalValues(description, target);

    assertEquals(3, target.size());
    assertTrue("'val1 not found'", target.contains("val1"));
    assertTrue("'val2 not found'", target.contains("val2"));
    assertTrue("'val3 not found'", target.contains("val3"));
  }

  @Test
  public void testExtractNumericalRange() {
    String description = "-2.06,12.32";
    double[] range = DescriptionUtils.extractNumericalRange(description);
    assertEquals(-2.06, range[0], EPSILON);
    assertEquals(12.32, range[1], EPSILON);
  }

}
