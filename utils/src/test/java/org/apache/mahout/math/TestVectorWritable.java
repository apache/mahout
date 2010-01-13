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

package org.apache.mahout.math;

import junit.framework.TestCase;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;

public class TestVectorWritable extends TestCase {

  private static final int cardinality = 10;

  private static void doTest(Vector vector) throws Exception {
    for (int i = 0; i < cardinality; i++) {
      vector.set(i, i);
    }
    DataOutputBuffer out = new DataOutputBuffer();
    VectorWritable v = new VectorWritable();
    v.set(vector);
    v.write(out);
    out.close();

    DataInputStream in = new DataInputStream(new ByteArrayInputStream(out.getData()));
    v = new VectorWritable();
    v.setConf(new Configuration());
    v.readFields(in);
    in.close();

    assertEquals(cardinality, vector.size());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double) i, vector.get(i));
    }

    in = new DataInputStream(new ByteArrayInputStream(out.getData()));
    v = new VectorWritable();
    v.setConf(new Configuration());
    v.readFields(in);
    in.close();

    assertEquals(cardinality, vector.size());
    for (int i = 0; i < cardinality; i++) {
      assertEquals((double) i, vector.get(i));
    }
  }

  public void testVectors() throws Exception {
    doTest(new SparseVector(cardinality));
    doTest(new DenseVector(cardinality));
    doTest(new VectorView(new SparseVector(cardinality + 1), 1, cardinality));
    doTest(new VectorView(new DenseVector(cardinality + 1), 1, cardinality));
  }
}
