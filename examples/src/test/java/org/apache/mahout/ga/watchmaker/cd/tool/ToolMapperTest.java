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

import junit.framework.TestCase;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.utils.DummyOutputCollector;

import java.util.List;

public class ToolMapperTest extends TestCase {

  public void testExtractAttributes() throws Exception {
    LongWritable key = new LongWritable();
    Text value = new Text();
    DummyOutputCollector<LongWritable, Text> output = new DummyOutputCollector<LongWritable, Text>();

    ToolMapper mapper = new ToolMapper();

    // no attribute is ignored
    String dataline = "A1, A2, A3, A4, A5, A6";
    char[] descriptors = { 'N', 'N', 'C', 'C', 'N', 'N' };

    mapper.configure(descriptors);
    value.set(dataline);
    mapper.map(key, value, output, null);

    for (int index = 0; index < 6; index++) {
      List<Text> values = output.getValue(String.valueOf(index));
      assertEquals("should extract one value per attribute", 1, values.size());
      assertEquals("Bad extracted value", "A" + (index + 1), values.get(0)
          .toString());
    }
  }

  public void testExtractIgnoredAttributes() throws Exception {
    LongWritable key = new LongWritable();
    Text value = new Text();
    DummyOutputCollector<LongWritable, Text> output = new DummyOutputCollector<LongWritable, Text>();

    ToolMapper mapper = new ToolMapper();

    // no attribute is ignored
    String dataline = "A1, I, A3, I, I, A6";
    char[] descriptors = { 'N', 'I', 'C', 'I', 'I', 'N' };

    mapper.configure(descriptors);
    value.set(dataline);
    mapper.map(key, value, output, null);

    for (int index = 0; index < 6; index++) {
      List<Text> values = output.getValue(String.valueOf(index));
      if (index == 1 || index == 3 || index == 4) {
        // this attribute should be ignored
        assertNull("Attribute (" + index + ") should be ignored", values);
      } else {
        assertEquals("should extract one value per attribute", 1, values.size());
        assertEquals("Bad extracted value", "A" + (index + 1), values.get(0)
            .toString());
      }
    }
  }
}
