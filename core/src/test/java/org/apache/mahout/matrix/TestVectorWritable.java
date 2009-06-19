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

package org.apache.mahout.matrix;

import junit.framework.TestCase;
import org.apache.hadoop.io.DataOutputBuffer;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;

public class TestVectorWritable extends TestCase {

  private static final int cardinality = 10;

  private static void doTest(Vector writable) throws Exception {
    for (int i = 0; i < cardinality; i++) {
      writable.set(i, i);
    }
    DataOutputBuffer out = new DataOutputBuffer();
    writable.write(out);
    out.close();

    DataInputStream in = new DataInputStream(new ByteArrayInputStream(out
        .getData()));
    writable.readFields(in);
    in.close();

    assertEquals(cardinality, writable.size());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double) i, writable.get(i));
    }

    in = new DataInputStream(new ByteArrayInputStream(out.getData()));
    writable.readFields(in);
    in.close();

    assertEquals(cardinality, writable.size());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double) i, writable.get(i));
    }
  }

  public void testVectors() throws Exception {
    doTest(new SparseVector(cardinality));
    doTest(new DenseVector(cardinality));
    doTest(new VectorView(new SparseVector(cardinality + 1), 1, cardinality));
    doTest(new VectorView(new DenseVector(cardinality + 1), 1, cardinality));
  }
}
