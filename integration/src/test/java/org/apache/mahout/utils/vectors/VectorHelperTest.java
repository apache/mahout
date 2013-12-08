/*
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

package org.apache.mahout.utils.vectors;

import com.google.common.collect.Iterables;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public final class VectorHelperTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;

  private Path inputPathOne;
  private Path inputPathTwo;

  private Configuration conf;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = getConfiguration();

    inputPathOne = getTestTempFilePath("documents/docs-one.file");
    FileSystem fs = FileSystem.get(inputPathOne.toUri(), conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPathOne, Text.class, IntWritable.class);
    try {
      Random rd = RandomUtils.getRandom();
      for (int i = 0; i < NUM_DOCS; i++) {
        // Make all indices higher than dictionary size
        writer.append(new Text("Document::ID::" + i), new IntWritable(NUM_DOCS + rd.nextInt(NUM_DOCS)));
      }
    } finally {
      Closeables.close(writer, false);
    }

    inputPathTwo = getTestTempFilePath("documents/docs-two.file");
    fs = FileSystem.get(inputPathTwo.toUri(), conf);
    writer = new SequenceFile.Writer(fs, conf, inputPathTwo, Text.class, IntWritable.class);
    try {
      Random rd = RandomUtils.getRandom();
      for (int i = 0; i < NUM_DOCS; i++) {
        // Keep indices within number of documents
        writer.append(new Text("Document::ID::" + i), new IntWritable(rd.nextInt(NUM_DOCS)));
      }
    } finally {
      Closeables.close(writer, false);
    }
  }

  @Test
  public void testJsonFormatting() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 3.1);
    v.set(4, 1.0);
    v.set(6, 8.1);
    v.set(7, -100);
    v.set(9, 12.2);
    String UNUSED = "UNUSED";
    String[] dictionary = {
        UNUSED, UNUSED, "two", UNUSED, "four", UNUSED, "six", "seven", UNUSED, "nine"
    };

    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1}",
        VectorHelper.vectorToJson(v, dictionary, 3, true));
    assertEquals("unsorted form incorrect: ", "{two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 2, false));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 4, true));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1,two:3.1,four:1.0,seven:-100.0}",
        VectorHelper.vectorToJson(v, dictionary, 5, true));
    assertEquals("sorted json form incorrect: ", "{nine:12.2,six:8.1}",
        VectorHelper.vectorToJson(v, dictionary, 2, true));
    assertEquals("unsorted form incorrect: ", "{two:3.1,four:1.0}",
        VectorHelper.vectorToJson(v, dictionary, 2, false));
  }

  @Test
  public void testTopEntries() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 3.1);
    v.set(4, 1.0);
    v.set(6, 8.1);
    v.set(7, -100);
    v.set(9, 12.2);
    v.set(1, 0.0);
    v.set(3, 0.0);
    v.set(8, 2.7);
    // check if sizeOFNonZeroElementsInVector = maxEntries
    assertEquals(6, VectorHelper.topEntries(v, 6).size());
    // check if sizeOfNonZeroElementsInVector < maxEntries
    assertTrue(VectorHelper.topEntries(v, 9).size() < 9);
    // check if sizeOfNonZeroElementsInVector > maxEntries
    assertTrue(VectorHelper.topEntries(v, 5).size() < Iterables.size(v.nonZeroes()));
  }

  @Test
  public void testTopEntriesWhenAllZeros() throws Exception {
    Vector v = new SequentialAccessSparseVector(10);
    v.set(2, 0.0);
    v.set(4, 0.0);
    v.set(6, 0.0);
    v.set(7, 0);
    v.set(9, 0.0);
    v.set(1, 0.0);
    v.set(3, 0.0);
    v.set(8, 0.0);
    assertEquals(0, VectorHelper.topEntries(v, 6).size());
  }

  @Test
  public void testLoadTermDictionary() throws Exception {
    // With indices higher than dictionary size
    VectorHelper.loadTermDictionary(conf, inputPathOne.toString());
    // With dictionary size higher than indices
    VectorHelper.loadTermDictionary(conf, inputPathTwo.toString());
  }
}
