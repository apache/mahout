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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * <p>Encodes signed and unsigned values using a common variable-length
 * scheme, found for example in
 * <a href="http://code.google.com/apis/protocolbuffers/docs/encoding.html">
 * Google's Protocol Buffers</a>. It uses fewer bytes to encode smaller values,
 * but will use slightly more bytes to encode large values.</p>
 *
 * <p>Signed values are further encoded using so-called zig-zag encoding
 * in order to make them "compatible" with variable-length encoding.</p>
 */
public final class Varint {

  public static final long MIN_SIGNED_VAR_LONG = -(1L << 62);
  public static final long MAX_SIGNED_VAR_LONG = (1L << 62) - 1;
  public static final int MIN_SIGNED_VAR_INT = -(1 << 30);
  public static final int MAX_SIGNED_VAR_INT = (1 << 30) - 1;

  private Varint() {
  }

  /**
   * Encodes a value using the variable-length encoding from
   * <a href="http://code.google.com/apis/protocolbuffers/docs/encoding.html">
   * Google Protocol Buffers</a>. It uses zig-zag encoding to efficiently
   * encode signed values. If values are known to be nonnegative,
   * {@link #writeUnsignedVarLong(long, DataOutput)} should be used.
   *
   * @param value value to encode
   * @param out to write bytes to
   * @throws IOException if {@link DataOutput} throws {@link IOException}
   * @throws IllegalArgumentException if value is not between {@link #MIN_SIGNED_VAR_LONG}
   *  and {@link #MAX_SIGNED_VAR_LONG} inclusive
   */
  public static void writeSignedVarLong(long value, DataOutput out) throws IOException {
    if (value < MIN_SIGNED_VAR_LONG || value > MAX_SIGNED_VAR_LONG) {
      throw new IllegalArgumentException("Can't encode value as signed: " + value);
    }
    // Great trick from http://code.google.com/apis/protocolbuffers/docs/encoding.html#types
    writeUnsignedVarLong((value << 1) ^ (value >> 63), out);
  }

  /**
   * Encodes a value using the variable-length encoding from
   * <a href="http://code.google.com/apis/protocolbuffers/docs/encoding.html">
   * Google Protocol Buffers</a>. Zig-zag is not used, so input must not be negative.
   * If values can be negative, use {@link #writeSignedVarLong(long, DataOutput)}
   * instead.
   *
   * @param value value to encode
   * @param out to write bytes to
   * @throws IOException if {@link DataOutput} throws {@link IOException}
   * @throws IllegalArgumentException if value is negative
   */
  public static void writeUnsignedVarLong(long value, DataOutput out) throws IOException {
    if (value < 0L) {
      throw new IllegalArgumentException("Can't encode negative value: " + value);
    }
    while ((value & 0xFFFFFFFFFFFFFF80L) != 0L) {
      out.writeByte(((int) value & 0x7F) | 0x80);
      value >>>= 7;
    }
    out.writeByte((int) value & 0x7F);
  }

  /**
   * See {@link #writeSignedVarLong(long, DataOutput)}
   */
  public static void writeSignedVarInt(int value, DataOutput out) throws IOException {
    if (value < MIN_SIGNED_VAR_INT || value > MAX_SIGNED_VAR_INT) {
      throw new IllegalArgumentException("Can't encode value as signed: " + value);
    }
    // Great trick from http://code.google.com/apis/protocolbuffers/docs/encoding.html#types
    writeUnsignedVarInt((value << 1) ^ (value >> 31), out);
  }

  /**
   * See {@link #writeUnsignedVarLong(long, DataOutput)}
   */
  public static void writeUnsignedVarInt(int value, DataOutput out) throws IOException {
    if (value < 0) {
      throw new IllegalArgumentException("Can't encode negative value: " + value);
    }
    while ((value & 0xFFFFFF80) != 0L) {
      out.writeByte((value & 0x7F) | 0x80);
      value >>>= 7;
    }
    out.writeByte(value & 0x7F);
  }

  /**
   * See {@link #writeSignedVarLong(long, DataOutput)}.
   *
   * @param in to read bytes from
   * @return decode value
   * @throws IOException if {@link DataInput} throws {@link IOException}
   * @throws IllegalArgumentException if variable-length value does not terminate
   *  after 8 bytes have been read
   */
  public static long readSignedVarLong(DataInput in) throws IOException {
    long raw = readUnsignedVarLong(in);
    // This undoes the trick in writeSignedVarLong()
    return (((raw << 63) >> 63) ^ raw) >> 1;
  }

  /**
   * See {@link #writeUnsignedVarLong(long, DataOutput)}.
   *
   * @param in to read bytes from
   * @return decode value
   * @throws IOException if {@link DataInput} throws {@link IOException}
   * @throws IllegalArgumentException if variable-length value does not terminate
   *  after 8 bytes have been read
   */
  public static long readUnsignedVarLong(DataInput in) throws IOException {
    long value = 0L;
    int i = 0;
    long b;
    while (((b = in.readByte()) & 0x80L) != 0) {
      value |= (b & 0x7F) << i;
      i += 7;
      if (i > 56) {
        throw new IllegalArgumentException("Variable length quantity is too long");
      }
    }
    return value | (b << i);
  }

  /**
   * See {@link #readSignedVarLong(DataInput)}
   */
  public static int readSignedVarInt(DataInput in) throws IOException {
    int raw = readUnsignedVarInt(in);
    // This undoes the trick in writeSignedVarInt()
    return (((raw << 31) >> 31) ^ raw) >> 1;
  }

  /**
   * See {@link #readUnsignedVarLong(DataInput)}
   */
  public static int readUnsignedVarInt(DataInput in) throws IOException {
    int value = 0;
    int i = 0;
    int b;
    while (((b = in.readByte()) & 0x80) != 0) {
      value |= (b & 0x7F) << i;
      i += 7;
      if (i > 28) {
        throw new IllegalArgumentException("Variable length quantity is too long");
      }
    }
    return value | (b << i);
  }

}
