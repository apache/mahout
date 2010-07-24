/*
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

package org.apache.mahout.classifier;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * This is a very fast, non-cryptographic hash suitable for general hash-based
 * lookup.  See http://murmurhash.googlepages.com/ for more details.
 * <p/>
 * <p>The C version of MurmurHash 2.0 found at that site was ported
 * to Java by Andrzej Bialecki (ab at getopt org).</p>
 */
public class MurmurHash {
  /**
   * Hashes bytes in an array.
   * @param data The bytes to hash.
   * @param seed The seed for the hash.
   * @return The 32 bit hash of the bytes in question.
   */
  public static int hash(byte[] data, int seed) {
    return hash(ByteBuffer.wrap(data), seed);
  }

  /**
   * Hashes bytes in part of an array.
   * @param data    The data to hash.
   * @param offset  Where to start munging.
   * @param length  How many bytes to process.
   * @param seed    The seed to start with.
   * @return        The 32-bit hash of the data in question.
   */
  public static int hash(byte[] data, int offset, int length, int seed) {
    return hash(ByteBuffer.wrap(data, offset, length), seed);
  }

  /**
   * Hashes the bytes in a buffer from the current position to the limit.
   * @param buf    The bytes to hash.
   * @param seed   The seed for the hash.
   * @return       The 32 bit murmur hash of the bytes in the buffer.
   */
  public static int hash(ByteBuffer buf, int seed) {
    // save byte order for later restoration
    ByteOrder byteOrder = buf.order();
    buf.order(ByteOrder.LITTLE_ENDIAN);

    int m = 0x5bd1e995;
    int r = 24;

    int h = seed ^ buf.remaining();

    int k;
    while (buf.remaining() >= 4) {
      k = buf.getInt();

      k *= m;
      k ^= k >>> r;
      k *= m;

      h *= m;
      h ^= k;
    }

    if (buf.remaining() > 0) {
      ByteBuffer finish = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
      // for big-endian version, use this first:
      // finish.position(4-buf.remaining());
      finish.put(buf).rewind();
      h ^= finish.getInt();
      h *= m;
    }

    h ^= h >>> 13;
    h *= m;
    h ^= h >>> 15;

    buf.order(byteOrder);
    return h;
  }


  public static long hash64A(byte[] data, int seed) {
    return hash64A(ByteBuffer.wrap(data), seed);
  }

  public static long hash64A(byte[] data, int offset, int length, int seed) {
    return hash64A(ByteBuffer.wrap(data, offset, length), seed);
  }

  public static long hash64A(ByteBuffer buf, int seed) {
    ByteOrder byteOrder = buf.order();
    buf.order(ByteOrder.LITTLE_ENDIAN);

    long m = 0xc6a4a7935bd1e995L;
    int r = 47;

    long h = seed ^ (buf.remaining() * m);

    long k;
    while (buf.remaining() >= 8) {
      k = buf.getLong();

      k *= m;
      k ^= k >>> r;
      k *= m;

      h ^= k;
      h *= m;
    }

    if (buf.remaining() > 0) {
      ByteBuffer finish = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      // for big-endian version, do this first:
      // finish.position(8-buf.remaining());
      finish.put(buf).rewind();
      h ^= finish.getLong();
      h *= m;
    }

    h ^= h >>> r;
    h *= m;
    h ^= h >>> r;

    buf.order(byteOrder);
    return h;
  }

  @Deprecated
  public static long hashLong(byte[] bytes, int seed) {
    return (((long) hash(bytes, seed ^ 120705477)) << 32) + hash(bytes, seed ^ 226137830);
  }

  @Deprecated
  public static int hash_original(byte[] data, int seed) {
    int m = 0x5bd1e995;
    int r = 24;

    int h = seed ^ data.length;

    int len = data.length;
    int len_4 = len >> 2;

    int k;
    for (int i = 0; i < len_4; i++) {
      int i_4 = i << 2;
      k = data[i_4];
      k |= data[i_4 + 1] << 8;
      k |= data[i_4 + 2] << 16;
      k |= data[i_4 + 3] << 24;

      k *= m;
      k ^= k >>> r;
      k *= m;

      h *= m;
      h ^= k;
    }

    int len_m = len_4 << 2;
    int left = len - len_m;

    switch (left) {
      case 3:
        h ^= (int) data[len_m + 2] << 16;
      case 2:
        h ^= (int) data[len_m + 1] << 8;
      case 1:
        h ^= (int) data[len_m];
        h *= m;
      default:
    }

    h ^= h >>> 13;
    h *= m;
    h ^= h >>> 15;

    return h;
  }

  @Deprecated
  public static long hash64A_original(byte[] data, int seed) {
    long m = 0xc6a4a7935bd1e995L;
    int r = 47;

    int len = data.length;
    int len_8 = len >> 3;

    long h = seed ^ (len * m);

    long k;
    for (int i = 0; i < len_8; i++) {
      int i_8 = i << 3;
      k = (data[i_8 + 7] & 0xffL) << 56;
      k |= (data[i_8 + 6] & 0xffL) << 48;
      k |= (data[i_8 + 5] & 0xffL) << 40;
      k |= (data[i_8 + 4] & 0xffL) << 32;
      k |= (data[i_8 + 3] & 0xffL) << 24;
      k |= (data[i_8 + 2] & 0xffL) << 16;
      k |= (data[i_8 + 1] & 0xffL) << 8;
      k |= (data[i_8] & 0xffL);

      k *= m;
      k ^= k >>> r;
      k *= m;

      h ^= k;
      h *= m;
    }

    int len_m = len_8 << 3;
    int left = len - len_m;

    switch (left) {
      case 7:
        h ^= (data[len_m + 6] & 0xffL) << 48;
      case 6:
        h ^= (data[len_m + 5] & 0xffL) << 40;
      case 5:
        h ^= (data[len_m + 4] & 0xffL) << 32;
      case 4:
        h ^= (data[len_m + 3] & 0xffL) << 24;
      case 3:
        h ^= (data[len_m + 2] & 0xffL) << 16;
      case 2:
        h ^= (data[len_m + 1] & 0xffL) << 8;
      case 1:
        h ^= data[len_m] & 0xffL;
        h *= m;
      default:
    }

    h ^= h >>> r;
    h *= m;
    h ^= h >>> r;

    return h;
  }


  /* Testing ...
 static int NUM = 1000;

 public static void main(String[] args) {
   byte[] bytes = new byte[4];
   for (int i = 0; i < NUM; i++) {
     bytes[0] = (byte)(i & 0xff);
     bytes[1] = (byte)((i & 0xff00) >> 8);
     bytes[2] = (byte)((i & 0xff0000) >> 16);
     bytes[3] = (byte)((i & 0xff000000) >> 24);
     System.out.println(Integer.toHexString(i) + " " + Integer.toHexString(hash(bytes, 1)));
   }
 } */
}
