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

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;

/**
 * Tests {@link Varint}.
 */
public final class VarintTest extends MahoutTestCase {

  @Test
  public void testUnsignedLong() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    Varint.writeUnsignedVarLong(0L, out);
    for (long i = 1L; i > 0L && i <= (1L << 62); i <<= 1) {
      Varint.writeUnsignedVarLong(i-1, out);
      Varint.writeUnsignedVarLong(i, out);
    }
    Varint.writeUnsignedVarLong(Long.MAX_VALUE, out);

    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    assertEquals(0L, Varint.readUnsignedVarLong(in));
    for (long i = 1L; i > 0L && i <= (1L << 62); i <<= 1) {
      assertEquals(i-1, Varint.readUnsignedVarLong(in));
      assertEquals(i, Varint.readUnsignedVarLong(in));
    }
    assertEquals(Long.MAX_VALUE, Varint.readUnsignedVarLong(in));
  }

  @Test
  public void testSignedPositiveLong() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    Varint.writeSignedVarLong(0L, out);
    for (long i = 1L; i <= (1L << 61); i <<= 1) {
      Varint.writeSignedVarLong(i-1, out);
      Varint.writeSignedVarLong(i, out);
    }
    Varint.writeSignedVarLong((1L << 62) - 1, out);
    Varint.writeSignedVarLong((1L << 62), out);
    Varint.writeSignedVarLong(Long.MAX_VALUE, out);

    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    assertEquals(0L, Varint.readSignedVarLong(in));
    for (long i = 1L; i <= (1L << 61); i <<= 1) {
      assertEquals(i-1, Varint.readSignedVarLong(in));
      assertEquals(i, Varint.readSignedVarLong(in));
    }
    assertEquals((1L << 62) - 1, Varint.readSignedVarLong(in));
    assertEquals((1L << 62), Varint.readSignedVarLong(in));
    assertEquals(Long.MAX_VALUE, Varint.readSignedVarLong(in));
  }

  @Test
  public void testSignedNegativeLong() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    for (long i = -1L; i >= -(1L << 62); i <<= 1) {
      Varint.writeSignedVarLong(i, out);
      Varint.writeSignedVarLong(i+1, out);
    }
    Varint.writeSignedVarLong(Long.MIN_VALUE, out);
    Varint.writeSignedVarLong(Long.MIN_VALUE+1, out);
    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    for (long i = -1L; i >= -(1L << 62); i <<= 1) {
      assertEquals(i, Varint.readSignedVarLong(in));
      assertEquals(i+1, Varint.readSignedVarLong(in));
    }
    assertEquals(Long.MIN_VALUE, Varint.readSignedVarLong(in));
    assertEquals(Long.MIN_VALUE+1, Varint.readSignedVarLong(in));
  }

  @Test
  public void testUnsignedInt() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    Varint.writeUnsignedVarInt(0, out);
    for (int i = 1; i > 0 && i <= (1 << 30); i <<= 1) {
      Varint.writeUnsignedVarLong(i-1, out);
      Varint.writeUnsignedVarLong(i, out);
    }
    Varint.writeUnsignedVarLong(Integer.MAX_VALUE, out);

    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    assertEquals(0, Varint.readUnsignedVarInt(in));
    for (int i = 1; i > 0 && i <= (1 << 30); i <<= 1) {
      assertEquals(i-1, Varint.readUnsignedVarInt(in));
      assertEquals(i, Varint.readUnsignedVarInt(in));
    }
    assertEquals(Integer.MAX_VALUE, Varint.readUnsignedVarInt(in));
  }

  @Test
  public void testSignedPositiveInt() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    Varint.writeSignedVarInt(0, out);
    for (int i = 1; i <= (1 << 29); i <<= 1) {
      Varint.writeSignedVarLong(i-1, out);
      Varint.writeSignedVarLong(i, out);
    }
    Varint.writeSignedVarInt((1 << 30) - 1, out);
    Varint.writeSignedVarInt((1 << 30), out);
    Varint.writeSignedVarInt(Integer.MAX_VALUE, out);

    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    assertEquals(0, Varint.readSignedVarInt(in));
    for (int i = 1; i <= (1 << 29); i <<= 1) {
      assertEquals(i-1, Varint.readSignedVarInt(in));
      assertEquals(i, Varint.readSignedVarInt(in));
    }
    assertEquals((1L << 30) - 1, Varint.readSignedVarInt(in));
    assertEquals((1L << 30), Varint.readSignedVarInt(in));
    assertEquals(Integer.MAX_VALUE, Varint.readSignedVarInt(in));
  }

  @Test
  public void testSignedNegativeInt() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    for (int i = -1; i >= -(1 << 30); i <<= 1) {
      Varint.writeSignedVarInt(i, out);
      Varint.writeSignedVarInt(i+1, out);
    }
    Varint.writeSignedVarInt(Integer.MIN_VALUE, out);
    Varint.writeSignedVarInt(Integer.MIN_VALUE+1, out);
    DataInput in = new DataInputStream(new ByteArrayInputStream(baos.toByteArray()));
    for (int i = -1; i >= -(1 << 30); i <<= 1) {
      assertEquals(i, Varint.readSignedVarInt(in));
      assertEquals(i+1, Varint.readSignedVarInt(in));
    }
    assertEquals(Integer.MIN_VALUE, Varint.readSignedVarInt(in));
    assertEquals(Integer.MIN_VALUE+1, Varint.readSignedVarInt(in));
  }

  @Test
  public void testUnsignedSize() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    int expectedSize = 0;
    for (int exponent = 0; exponent <= 62; exponent++) {
      Varint.writeUnsignedVarLong(1L << exponent, out);
      expectedSize += 1 + exponent / 7;
      assertEquals(expectedSize, baos.size());
    }
  }

  @Test
  public void testSignedSize() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(baos);
    int expectedSize = 0;
    for (int exponent = 0; exponent <= 61; exponent++) {
      Varint.writeSignedVarLong(1L << exponent, out);
      expectedSize += 1 + ((exponent + 1) / 7);
      assertEquals(expectedSize, baos.size());
    }
    for (int exponent = 0; exponent <= 61; exponent++) {
      Varint.writeSignedVarLong(-(1L << exponent)-1, out);
      expectedSize += 1 + ((exponent + 1) / 7);
      assertEquals(expectedSize, baos.size());
    }
  }

}
