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

import com.google.common.io.Closeables;
import org.apache.hadoop.io.Writable;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public final class VectorWritableTest extends MahoutTestCase {

  @Test
  public void testSequentialAccessSparseVectorWritable() throws Exception {
    Vector v = new SequentialAccessSparseVector(5);
    v.set(1, 3.0);
    v.set(3, 5.0);
    doTestVectorWritableEquals(v);
  }

  @Test
  public void testRandomAccessSparseVectorWritable() throws Exception {
    Vector v = new RandomAccessSparseVector(5);
    v.set(1, 3.0);
    v.set(3, 5.0);
    doTestVectorWritableEquals(v);
  }

  @Test
  public void testDenseVectorWritable() throws Exception {
    Vector v = new DenseVector(5);
    v.set(1, 3.0);
    v.set(3, 5.0);
    doTestVectorWritableEquals(v);
  }

  @Test
  public void testNamedVectorWritable() throws Exception {
    Vector v = new DenseVector(5);
    v = new NamedVector(v, "Victor");
    v.set(1, 3.0);
    v.set(3, 5.0);
    doTestVectorWritableEquals(v);
  }

  private static void doTestVectorWritableEquals(Vector v) throws IOException {
    Writable vectorWritable = new VectorWritable(v);
    VectorWritable vectorWritable2 = new VectorWritable();
    writeAndRead(vectorWritable, vectorWritable2);
    Vector v2 = vectorWritable2.get();
    if (v instanceof NamedVector) {
    	assertTrue(v2 instanceof NamedVector);
    	NamedVector nv = (NamedVector) v;
    	NamedVector nv2 = (NamedVector) v2;
    	assertEquals(nv.getName(), nv2.getName());
    	assertEquals("Victor", nv.getName());
    }
    assertEquals(v, v2);
  }

  private static void writeAndRead(Writable toWrite, Writable toRead) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutputStream dos = new DataOutputStream(baos);
    try {
      toWrite.write(dos);
    } finally {
      Closeables.closeQuietly(dos);
    }

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    DataInputStream dis = new DataInputStream(bais);
    try {
      toRead.readFields(dis);
    } finally {
      Closeables.closeQuietly(dis);
    }
  }


}
