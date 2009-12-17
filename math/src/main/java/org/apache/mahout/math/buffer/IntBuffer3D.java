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
import org.apache.mahout.math.list.IntArrayList;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class IntBuffer3D extends PersistentObject implements IntBuffer3DConsumer {

  private final IntBuffer3DConsumer target;
  private final int[] xElements;
  private final int[] yElements;
  private final int[] zElements;

  // vars cached for speed
  private final IntArrayList xList;
  private final IntArrayList yList;
  private final IntArrayList zList;
  private final int capacity;
  private int size;

  /**
   * Constructs and returns a new buffer with the given target.
   *
   * @param target   the target to flush to.
   * @param capacity the number of points the buffer shall be capable of holding before overflowing and flushing to the
   *                 target.
   */
  public IntBuffer3D(IntBuffer3DConsumer target, int capacity) {
    this.target = target;
    this.capacity = capacity;
    this.xElements = new int[capacity];
    this.yElements = new int[capacity];
    this.zElements = new int[capacity];
    this.xList = new IntArrayList(xElements);
    this.yList = new IntArrayList(yElements);
    this.zList = new IntArrayList(zElements);
    this.size = 0;
  }

  /**
   * Adds the specified point (x,y,z) to the receiver.
   *
   * @param x the x-coordinate of the point to add.
   * @param y the y-coordinate of the point to add.
   * @param z the z-coordinate of the point to add.
   */
  public void add(int x, int y, int z) {
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
  public void addAllOf(IntArrayList xElements, IntArrayList yElements, IntArrayList zElements) {
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
