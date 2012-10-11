package org.apache.mahout.math.set;

/**
 * Computes hashes of primitive values.  Providing these as statics allows the templated code
 * to compute hashes of sets.
 */
public class HashUtils {
  public static int hash(byte x) {
    return x;
  }

  public static int hash(short x) {
    return x;
  }

  public static int hash(char x) {
    return x;
  }

  public static int hash(int x) {
    return x;
  }

  public static int hash(float x) {
    return Float.floatToIntBits(x) >>> 3 + Float.floatToIntBits((float) (Math.PI * x));
  }

  public static int hash(double x) {
    return hash(17 * Double.doubleToLongBits(x));
  }

  public static int hash(long x) {
    return (int) ((x * 11) >>> 32 ^ x);
  }
}
