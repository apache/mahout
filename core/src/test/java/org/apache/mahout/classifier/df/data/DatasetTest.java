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

package org.apache.mahout.classifier.df.data;

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
import org.junit.Test;

public final class DatasetTest extends MahoutTestCase {

  private static final int NUM_ATTRIBUTES = 10;

  private static Dataset readDataset(byte[] bytes) throws IOException {
    ByteArrayInputStream byteInStream = new ByteArrayInputStream(bytes);
    DataInput in = new DataInputStream(byteInStream);
    return Dataset.read(in);
  }

  @Test
  public void testWritable() throws Exception {

    Random rng = RandomUtils.getRandom();
    ByteArrayOutputStream byteOutStream = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(byteOutStream);

    int n = 10;
    for (int nloop = 0; nloop < n; nloop++) {
      byteOutStream.reset();
      
      Dataset dataset = Utils.randomData(rng, NUM_ATTRIBUTES, false, 1).getDataset();
      
      dataset.write(out);
      
      assertEquals(dataset, readDataset(byteOutStream.toByteArray()));

      // regression
      byteOutStream.reset();
      
      dataset = Utils.randomData(rng, NUM_ATTRIBUTES, true, 1).getDataset();
      
      dataset.write(out);
      
      assertEquals(dataset, readDataset(byteOutStream.toByteArray()));
    }
  }
  
}
