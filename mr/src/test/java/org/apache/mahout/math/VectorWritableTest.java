/**
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license
 * agreements. See the NOTICE file distributed with this work for additional information regarding
 * copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.apache.mahout.math;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector.Element;
import org.junit.Test;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import com.google.common.io.Closeables;

public final class VectorWritableTest extends RandomizedTest {
  private static final int MAX_VECTOR_SIZE = 100;

  public void createRandom(Vector v) {
    int size = randomInt(v.size() - 1);
    for (int i = 0; i < size; ++i) {
      v.set(randomInt(v.size() - 1), randomDouble());
    }

    int zeros = Math.max(2, size / 4);
    for (Element e : v.nonZeroes()) {
      if (e.index() % zeros == 0) {
        e.set(0.0);
      }
    }
  }

  @Test
  @Repeat(iterations = 20)
  public void testViewSequentialAccessSparseVectorWritable() throws Exception {
    Vector v = new SequentialAccessSparseVector(MAX_VECTOR_SIZE);
    createRandom(v);
    Vector view = new VectorView(v, 0, v.size());
    doTestVectorWritableEquals(view);
  }

  @Test
  @Repeat(iterations = 20)
  public void testSequentialAccessSparseVectorWritable() throws Exception {
    Vector v = new SequentialAccessSparseVector(MAX_VECTOR_SIZE);
    createRandom(v);
    doTestVectorWritableEquals(v);
  }

  @Test
  @Repeat(iterations = 20)
  public void testRandomAccessSparseVectorWritable() throws Exception {
    Vector v = new RandomAccessSparseVector(MAX_VECTOR_SIZE);
    createRandom(v);
    doTestVectorWritableEquals(v);
  }

  @Test
  @Repeat(iterations = 20)
  public void testDenseVectorWritable() throws Exception {
    Vector v = new DenseVector(MAX_VECTOR_SIZE);
    createRandom(v);
    doTestVectorWritableEquals(v);
  }

  @Test
  @Repeat(iterations = 20)
  public void testNamedVectorWritable() throws Exception {
    Vector v = new DenseVector(MAX_VECTOR_SIZE);
    v = new NamedVector(v, "Victor");
    createRandom(v);
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
      Closeables.close(dos, false);
    }

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    DataInputStream dis = new DataInputStream(bais);
    try {
      toRead.readFields(dis);
    } finally {
      Closeables.close(dos, true);
    }
  }
}
