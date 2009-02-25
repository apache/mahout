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

package org.apache.mahout.cf.taste.impl.common;

/**
 * A simplified and streamlined version of {@link java.util.BitSet}.
 */
final class BitSet {

  private final long[] bits;

  BitSet(int numBits) {
    int numLongs = numBits >>> 6;
    if (numBits % 64 != 0) {
      numLongs++;
    }
    bits = new long[numLongs];
  }

  boolean get(int index) {
    // skipping range check for speed
    int offset = index >>> 6;
    return (bits[offset] & (1L << (index - (offset << 6)))) != 0L;
  }

  void set(int index) {
    // skipping range check for speed
    int offset = index >>> 6;
    bits[offset] |= (1L << (index - (offset << 6)));
  }

  void clear(int index) {
    // skipping range check for speed
    int offset = index >>> 6;
    bits[offset] &= ~(1L << (index - (offset << 6)));
  }

  void clear() {
    for (int i = 0; i < bits.length; i++) {
      bits[i] = 0L;
    }
  }

}
