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

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.examples.MahoutTestCase;
import org.junit.Test;

public final class ToolMapperTest extends MahoutTestCase {

  @Test
  public void testExtractAttributes() throws Exception {
    LongWritable key = new LongWritable();
    Text value = new Text();
    Configuration conf = new Configuration();
    ToolMapper mapper = new ToolMapper();
    DummyRecordWriter<LongWritable, Text> writer = new DummyRecordWriter<LongWritable, Text>();
    Mapper<LongWritable, Text, LongWritable, Text>.Context context = DummyRecordWriter.build(mapper, conf, writer);

    // no attribute is ignored
    char[] descriptors = { 'N', 'N', 'C', 'C', 'N', 'N' };

    mapper.configure(descriptors);
    String dataline = "A1, A2, A3, A4, A5, A6";
    value.set(dataline);
    mapper.map(key, value, context);

    for (int index = 0; index < 6; index++) {
      List<Text> values = writer.getValue(new LongWritable(index));
      assertEquals("should extract one value per attribute", 1, values.size());
      assertEquals("Bad extracted value", "A" + (index + 1), values.get(0).toString());
    }
  }

  @Test
  public void testExtractIgnoredAttributes() throws Exception {
    LongWritable key = new LongWritable();
    Text value = new Text();
    ToolMapper mapper = new ToolMapper();
    Configuration conf = new Configuration();
    DummyRecordWriter<LongWritable, Text> writer = new DummyRecordWriter<LongWritable, Text>();
    Mapper<LongWritable, Text, LongWritable, Text>.Context context = DummyRecordWriter.build(mapper, conf, writer);

    // no attribute is ignored
    char[] descriptors = { 'N', 'I', 'C', 'I', 'I', 'N' };

    mapper.configure(descriptors);
    String dataline = "A1, I, A3, I, I, A6";
    value.set(dataline);
    mapper.map(key, value, context);

    for (int index = 0; index < 6; index++) {
      List<Text> values = writer.getValue(new LongWritable(index));
      if (index == 1 || index == 3 || index == 4) {
        // this attribute should be ignored
        assertNull("Attribute (" + index + ") should be ignored", values);
      } else {
        assertEquals("should extract one value per attribute", 1, values.size());
        assertEquals("Bad extracted value", "A" + (index + 1), values.get(0).toString());
      }
    }
  }
}
