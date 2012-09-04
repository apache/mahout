package org.apache.mahout.math;

/**
 * Marker interface for vectors that may cache their squared length.
 */
interface LengthCachingVector {
  public double getLengthSquared();

  /**
   * This is a very dangerous method to call.  Passing in a wrong value can
   * completely screw up distance computations and normalization.
   * @param d2  The new value for the squared length cache.
   */
  public void setLengthSquared(double d2);
}
