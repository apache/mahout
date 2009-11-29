/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.buffer;

import org.apache.mahout.matrix.PersistentObject;
import org.apache.mahout.matrix.list.ObjectArrayList;
/**
 * Fixed sized (non resizable) streaming buffer connected to a target <tt>ObjectBufferConsumer</tt> to which data is automatically flushed upon buffer overflow.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class ObjectBuffer extends PersistentObject implements ObjectBufferConsumer {

  private final ObjectBufferConsumer target;
  private final Object[] elements;

  // vars cached for speed
  private final ObjectArrayList list;
  private final int capacity;
  private int size;

  /**
   * Constructs and returns a new buffer with the given target.
   *
   * @param target   the target to flush to.
   * @param capacity the number of points the buffer shall be capable of holding before overflowing and flushing to the
   *                 target.
   */
  public ObjectBuffer(ObjectBufferConsumer target, int capacity) {
    this.target = target;
    this.capacity = capacity;
    this.elements = new Object[capacity];
    this.list = new ObjectArrayList(elements);
    this.size = 0;
  }

  /**
   * Adds the specified element to the receiver.
   *
   * @param element the element to add.
   */
  public void add(Object element) {
    if (this.size == this.capacity) {
      flush();
    }
    this.elements[size++] = element;
  }

  /**
   * Adds all elements of the specified list to the receiver.
   *
   * @param list the list of which all elements shall be added.
   */
  @Override
  public void addAllOf(ObjectArrayList list) {
    int listSize = list.size();
    if (this.size + listSize >= this.capacity) {
      flush();
    }
    this.target.addAllOf(list);
  }

  /** Sets the receiver's size to zero. In other words, forgets about any internally buffered elements. */
  public void clear() {
    this.size = 0;
  }

  /** Adds all internally buffered elements to the receiver's target, then resets the current buffer size to zero. */
  public void flush() {
    if (this.size > 0) {
      list.setSize(this.size);
      this.target.addAllOf(list);
      this.size = 0;
    }
  }
}
