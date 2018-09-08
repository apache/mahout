/*
 *  This code is public domain.
 *
 *  The MurmurHash3 algorithm was created by Austin Appleby and put into the public domain.
 *  See http://code.google.com/p/smhasher/
 *
 *  This java port was authored by
 *  Yonik Seeley and was placed into the public domain per
 *  https://github.com/yonik/java_util/blob/master/src/util/hash/MurmurHash3.java.
 */

package org.apache.mahout.math;

/**
 *  <p>
 *  This produces exactly the same hash values as the final C+
 + *  version of MurmurHash3 and is thus suitable for producing the same hash values across
 *  platforms.
 *  <p>
 *  The 32 bit x86 version of this hash should be the fastest variant for relatively short keys like ids.
 *  <p>
 *  Note - The x86 and x64 versions do _not_ produce the same results, as the
 *  algorithms are optimized for their respective platforms.
 *  <p>
 *  See also http://github.com/yonik/java_util for future updates to this file.
 */
public final class MurmurHash3 {

  private MurmurHash3() {}

  /** Returns the MurmurHash3_x86_32 hash. */
  public static int murmurhash3x8632(byte[] data, int offset, int len, int seed) {

    int c1 = 0xcc9e2d51;
    int c2 = 0x1b873593;

    int h1 = seed;
    int roundedEnd = offset + (len & 0xfffffffc);  // round down to 4 byte block

    for (int i = offset; i < roundedEnd; i += 4) {
      // little endian load order
      int k1 = (data[i] & 0xff) | ((data[i + 1] & 0xff) << 8) | ((data[i + 2] & 0xff) << 16) | (data[i + 3] << 24);
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >>> 17);  // ROTL32(k1,15);
      k1 *= c2;

      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);  // ROTL32(h1,13);
      h1 = h1 * 5 + 0xe6546b64;
    }

    // tail
    int k1 = 0;

    switch(len & 0x03) {
      case 3:
        k1 = (data[roundedEnd + 2] & 0xff) << 16;
        // fallthrough
      case 2:
        k1 |= (data[roundedEnd + 1] & 0xff) << 8;
        // fallthrough
      case 1:
        k1 |= data[roundedEnd] & 0xff;
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >>> 17);  // ROTL32(k1,15);
        k1 *= c2;
        h1 ^= k1;
      default:
    }

    // finalization
    h1 ^= len;

    // fmix(h1);
    h1 ^= h1 >>> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >>> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >>> 16;

    return h1;
  }

}
