/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.buffer;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.list.DoubleArrayList;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DoubleBuffer3D extends PersistentObject implements DoubleBuffer3DConsumer {

  private final DoubleBuffer3DConsumer target;
  private final double[] xElements;
  private final double[] yElements;
  private final double[] zElements;

  // vars cached for speed
  private final DoubleArrayList xList;
  private final DoubleArrayList yList;
  private final DoubleArrayList zList;
  private final int capacity;
  private int size;

  /**
   * Constructs and returns a new buffer with the given target.
   *
   * @param target   the target to flush to.
   * @param capacity the number of points the buffer shall be capable of holding before overflowing and flushing to the
   *                 target.
   */
  public DoubleBuffer3D(DoubleBuffer3DConsumer target, int capacity) {
    this.target = target;
    this.capacity = capacity;
    this.xElements = new double[capacity];
    this.yElements = new double[capacity];
    this.zElements = new double[capacity];
    this.xList = new DoubleArrayList(xElements);
    this.yList = new DoubleArrayList(yElements);
    this.zList = new DoubleArrayList(zElements);
    this.size = 0;
  }

  /**
   * Adds the specified point (x,y,z) to the receiver.
   *
   * @param x the x-coordinate of the point to add.
   * @param y the y-coordinate of the point to add.
   * @param z the z-coordinate of the point to add.
   */
  public void add(double x, double y, double z) {
    if (this.size == this.capacity) {
      flush();
    }
    this.xElements[this.size] = x;
    this.yElements[this.size] = y;
    this.zElements[this.size++] = z;
  }

  /**
   * Adds all specified (x,y,z) points to the receiver.
   *
   * @param xElements the x-coordinates of the points.
   * @param yElements the y-coordinates of the points.
   * @param zElements the y-coordinates of the points.
   */
  @Override
  public void addAllOf(DoubleArrayList xElements, DoubleArrayList yElements, DoubleArrayList zElements) {
    int listSize = xElements.size();
    if (this.size + listSize >= this.capacity) {
      flush();
    }
    this.target.addAllOf(xElements, yElements, zElements);
  }

  /** Sets the receiver's size to zero. In other words, forgets about any internally buffered elements. */
  public void clear() {
    this.size = 0;
  }

  /** Adds all internally buffered points to the receiver's target, then resets the current buffer size to zero. */
  public void flush() {
    if (this.size > 0) {
      xList.setSize(this.size);
      yList.setSize(this.size);
      zList.setSize(this.size);
      this.target.addAllOf(xList, yList, zList);
      this.size = 0;
    }
  }
}
