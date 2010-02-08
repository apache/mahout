/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.buffer;

import org.apache.mahout.math.list.DoubleArrayList;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface DoubleBuffer3DConsumer {

  /**
   * Adds all specified (x,y,z) points to the receiver.
   *
   * @param x the x-coordinates of the points to be added.
   * @param y the y-coordinates of the points to be added.
   * @param z the z-coordinates of the points to be added.
   */
  void addAllOf(DoubleArrayList x, DoubleArrayList y, DoubleArrayList z);
}
