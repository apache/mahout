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

package org.apache.mahout.df.mapreduce.inmem;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.mapreduce.inmem.InMemInputFormat.InMemInputSplit;

import junit.framework.TestCase;

public class InMemInputSplitTest extends TestCase {

  protected Random rng;

  protected ByteArrayOutputStream byteOutStream;
  protected DataOutput out;
  
  @Override
  protected void setUp() throws Exception {
    RandomUtils.useTestSeed();

    rng = RandomUtils.getRandom();

    byteOutStream = new ByteArrayOutputStream();
    out = new DataOutputStream(byteOutStream);
  }

  /**
   * Make sure that all the fields are processed correctly 
   * @throws IOException 
   *
   */
  public void testWritable() throws IOException {
    InMemInputSplit split = new InMemInputSplit(rng.nextInt(), rng.nextInt(1000), rng.nextLong());
    
    split.write(out);
    assertEquals(split, readSplit());
  }

  /**
   * test the case seed == null
   * @throws IOException 
   *
   */
  public void testNullSeed() throws IOException {
    InMemInputSplit split = new InMemInputSplit(rng.nextInt(), rng.nextInt(1000), null);
    
    split.write(out);
    assertEquals(split, readSplit());
  }
  
  protected InMemInputSplit readSplit() throws IOException {
    ByteArrayInputStream byteInStream = new ByteArrayInputStream(byteOutStream.toByteArray());
    DataInput in = new DataInputStream(byteInStream);
    return InMemInputSplit.read(in);
  }
}
