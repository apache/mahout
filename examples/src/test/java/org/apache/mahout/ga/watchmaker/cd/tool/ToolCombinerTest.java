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

import org.apache.hadoop.io.Text;
import org.apache.mahout.examples.MahoutTestCase;
import org.junit.Test;

import java.util.List;
import java.util.StringTokenizer;

public final class ToolCombinerTest extends MahoutTestCase {

  @Test
  public void testCreateDescriptionNumerical() throws Exception {
    ToolCombiner combiner = new ToolCombiner();

    char[] descriptors = { 'I', 'N', 'C' };
    combiner.configure(descriptors);

    List<Text> values = ToolReducerTest.asList("0", "10", "-32", "0.5", "-30");
    String descriptor = combiner.createDescription(1, values.iterator());

    assertEquals("-32.0,10.0", descriptor);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testCreateDescriptionIgnored() throws Exception {
    ToolCombiner combiner = new ToolCombiner();

    char[] descriptors = { 'I', 'N', 'C' };
    combiner.configure(descriptors);
    combiner.createDescription(0, null);
  }

  @Test
  public void testCreateDescriptionNominal() throws Exception {
    ToolCombiner combiner = new ToolCombiner();

    char[] descriptors = { 'I', 'N', 'C' };
    combiner.configure(descriptors);

    List<Text> values = ToolReducerTest.asList("val1", "val2", "val1", "val3", "val2");
    String descriptor = combiner.createDescription(2, values.iterator());

    StringTokenizer tokenizer = new StringTokenizer(descriptor, ",");
    int nbvalues = 0;
    while (tokenizer.hasMoreTokens()) {
      String value = tokenizer.nextToken().trim();
      if (!"val1".equals(value) && !"val2".equals(value) && !"val3".equals(value)) {
        fail("Incorrect value : " + value);
      }
      nbvalues++;
    }
    assertEquals(3, nbvalues);
  }

}
