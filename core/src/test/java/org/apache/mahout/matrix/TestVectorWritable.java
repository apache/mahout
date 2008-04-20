package org.apache.mahout.matrix;

import junit.framework.TestCase;
import org.apache.hadoop.io.DataOutputBuffer;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;

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

public class TestVectorWritable extends TestCase {

  private static final int cardinality = 10;

  public void test(VectorWritable writable) throws Exception {
    for (int i = 0; i < cardinality; i++) {
      writable.get().set(i, i);
    }
    DataOutputBuffer out = new DataOutputBuffer();
    writable.write(out);
    out.close();

    DataInputStream in = new DataInputStream(new ByteArrayInputStream(out.getData()));
    writable.readFields(in);
    in.close();

    assertEquals(cardinality, writable.get().cardinality());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double)i, writable.get().get(i));
    }

    // also make sure it creates the vector correct even if it is not set.
    writable.set(null);

    in = new DataInputStream(new ByteArrayInputStream(out.getData()));
    writable.readFields(in);
    in.close();

    assertEquals(cardinality, writable.get().cardinality());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double)i, writable.get().get(i));
    }


  }

  public void test() throws Exception {
    test(new SparseVectorWritable(new SparseVector(cardinality)));
    test(new DenseVectorWritable(new DenseVector(cardinality)));
  }
}
