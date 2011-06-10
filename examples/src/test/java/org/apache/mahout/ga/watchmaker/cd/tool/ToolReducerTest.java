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
import org.apache.hadoop.io.Text;
import org.apache.mahout.examples.MahoutTestCase;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public final class ToolReducerTest extends MahoutTestCase {

  @Test
  public void testCreateDescriptionNumerical() throws Exception {
    ToolReducer reducer = new ToolReducer();

    char[] descriptors = { 'I', 'N', 'C' };
    reducer.configure(descriptors);

    List<Text> values = asList("0,5", "-2,12", "-32,3",
        "0.5,25", "-30,20");
    String descriptor = ToolReducer.numericDescription(values.iterator());

    assertEquals("-32.0,25.0", descriptor);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testCreateDescriptionIgnored() throws Exception {
    ToolReducer reducer = new ToolReducer();

    char[] descriptors = { 'I', 'N', 'C' };
    reducer.configure(descriptors);
    reducer.combineDescriptions(0, null);
  }

  @Test
  public void testCreateDescriptionNominal() throws Exception {
    ToolReducer reducer = new ToolReducer();

    char[] descriptors = { 'I', 'N', 'C' };
    reducer.configure(descriptors);

    List<Text> values = asList("val1,val2", "val2,val3", "val1,val3", "val3",
        "val2,val4");
    List<String> expected = Arrays.asList("val1", "val2", "val3", "val4");

    String description = reducer.nominalDescription(values.iterator());

    Collection<String> actual = Lists.newArrayList();
    DescriptionUtils.extractNominalValues(description, actual);

    assertEquals(expected.size(), actual.size());
    assertTrue(expected.containsAll(actual));
  }

  static List<Text> asList(String... strings) {
    List<Text> values = Lists.newArrayList();

    for (String value : strings) {
      values.add(new Text(value));
    }
    return values;
  }
}
