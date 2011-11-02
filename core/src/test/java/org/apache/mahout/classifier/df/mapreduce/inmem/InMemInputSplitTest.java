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

package org.apache.mahout.classifier.df.mapreduce.inmem;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat.InMemInputSplit;
import org.junit.Before;
import org.junit.Test;

public final class InMemInputSplitTest extends MahoutTestCase {

  private Random rng;
  private ByteArrayOutputStream byteOutStream;
  private DataOutput out;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    rng = RandomUtils.getRandom();
    byteOutStream = new ByteArrayOutputStream();
    out = new DataOutputStream(byteOutStream);
  }

  /**
   * Make sure that all the fields are processed correctly
   */
  @Test
  public void testWritable() throws Exception {
    InMemInputSplit split = new InMemInputSplit(rng.nextInt(), rng.nextInt(1000), rng.nextLong());
    
    split.write(out);
    assertEquals(split, readSplit());
  }

  /**
   * test the case seed == null
   */
  @Test
  public void testNullSeed() throws Exception {
    InMemInputSplit split = new InMemInputSplit(rng.nextInt(), rng.nextInt(1000), null);
    
    split.write(out);
    assertEquals(split, readSplit());
  }
  
  private InMemInputSplit readSplit() throws IOException {
    ByteArrayInputStream byteInStream = new ByteArrayInputStream(byteOutStream.toByteArray());
    DataInput in = new DataInputStream(byteInStream);
    return InMemInputSplit.read(in);
  }
}
