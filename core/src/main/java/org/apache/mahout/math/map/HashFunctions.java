/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.map;


/**
 * Provides various hash functions.
 */
public final class HashFunctions {

  /**
   * Utility class pattern: all static members, no inheritance.
   */
  private HashFunctions() {
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(char value) {
    return value;
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(double value) {
    long bits = Double.doubleToLongBits(value);
    return (int) (bits ^ (bits >>> 32));

    //return (int) Double.doubleToLongBits(value*663608941.737);
    // this avoids excessive hashCollisions in the case values are of the form (1.0, 2.0, 3.0, ...)
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(float value) {
    return Float.floatToIntBits(value * 663608941.737f);
    // this avoids excessive hashCollisions in the case values are of the form (1.0, 2.0, 3.0, ...)
  }

  /**
   * Returns a hashcode for the specified value.
   * The hashcode computation is similar to the last step
   * of MurMurHash3.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(int value) {
    int h = value;
    h ^= h >>> 16;
    h *= 0x85ebca6b;
    h ^= h >>> 13;
    h *= 0xc2b2ae35;
    h ^= h >>> 16;
    return h;
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(long value) {
    return (int) (value ^ (value >> 32));
    /*
    value &= 0x7FFFFFFFFFFFFFFFL; // make it >=0 (0x7FFFFFFFFFFFFFFFL==Long.MAX_VALUE)
    int hashCode = 0;
    do hashCode = 31*hashCode + (int) (value%10);
    while ((value /= 10) > 0);

    return 28629151*hashCode; // spread even further; h*31^5
    */
  }

  /**
   * Returns a hashcode for the specified object.
   *
   * @return a hash code value for the specified object.
   */
  public static int hash(Object object) {
    return object == null ? 0 : object.hashCode();
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(short value) {
    return value;
  }

  /**
   * Returns a hashcode for the specified value.
   *
   * @return a hash code value for the specified value.
   */
  public static int hash(boolean value) {
    return value ? 1231 : 1237;
  }
}
