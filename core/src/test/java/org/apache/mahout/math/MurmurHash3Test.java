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

import org.junit.Test;

public final class MurmurHash3Test extends MahoutTestCase {

  private static final int[] ANSWERS =
      {0x0,0xcf9ce026,0x7b1ebceb,0x8a59e474,0xcf337f94,0x8b678f66,0x813ff5a2,0x1c2f4b2b,0xa6fcba77,0xe658f908,
          0x9f2656af,0x826b85ca,0xebb6ceca,0x24c4112c,0x66eff5b0,0xa9aca7d5,0xf7f04d03,0x9d781105,0x6dcde4f3,
          0x69edd8a8,0x5cdcd417,0x18d67f6,0xea040c90,0xdf70ea4a,0x8fb349e6,0x79a89b03,0x7ef9fc34,0x6017f692,
          0x5be02058,0x9e3986f9,0x8fa6dd28,0x6733b993,0x26230d32,0x92051d69,0x8d6f37f7,0xa1653103,0x8491c23f,
          0x2e8f59ce,0x5ae9461e,0xfe286e6,0x844e6959,0x87e9065d,0xe302e21c,0x1b3b3296,0xd29849c9,0x4e625f26,
          0xa8c35ac0,0x71335a06,0xfd256d8f,0x4e5eb258,0x4e2320d1,0xba2e9832,0xb00df8eb,0xbd87594d,0x83b6dce3,
          0xcf8646d0,0x7e79f2e2,0xd41fcd97,0x556a93,0x4419437b,0x39aa0e4e,0x43a57251,0x9430922f,0xd784b08f,
          0xa2772512,0xa2a6ee4b,0x9cb1abae,0xebd2bef0};

  @Test
  public void testCorrectValues() throws Exception {
    byte[] bytes = "Now is the time for all good men to come to the aid of their country".getBytes("UTF-8");
    int hash = 0;
    for (int i = 0; i < bytes.length; i++) {
      hash = hash * 31 + (bytes[i] & 0xff);
      bytes[i] = (byte) hash;
    }

    // test different offsets.
    for (int offset = 0; offset < 10; offset++) {
      byte[] arr = new byte[bytes.length + offset];
      System.arraycopy(bytes, 0, arr, offset, bytes.length);
      for (int len = 0; len < bytes.length; len++) {
        int h = MurmurHash3.murmurhash3x8632(arr, offset, len, len);
        assertEquals(ANSWERS[len], h);
      }
    }
  }

}