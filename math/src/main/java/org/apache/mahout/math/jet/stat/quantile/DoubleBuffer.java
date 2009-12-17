/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

import org.apache.mahout.math.list.DoubleArrayList;

/** A buffer holding <tt>double</tt> elements; internally used for computing approximate quantiles. */
class DoubleBuffer extends Buffer {

  protected DoubleArrayList values;
  protected boolean isSorted;

  /**
   * This method was created in VisualAge.
   *
   * @param k int
   */
  DoubleBuffer(int k) {
    super(k);
    this.values = new DoubleArrayList(0);
    this.isSorted = false;
  }

  /** Adds a value to the receiver. */
  public void add(double value) {
    if (!isAllocated) {
      allocate();
    } // lazy buffer allocation can safe memory.
    values.add(value);
    this.isSorted = false;
  }

  /** Adds a value to the receiver. */
  public void addAllOfFromTo(DoubleArrayList elements, int from, int to) {
    if (!isAllocated) {
      allocate();
    } // lazy buffer allocation can safe memory.
    values.addAllOfFromTo(elements, from, to);
    this.isSorted = false;
  }

  /** Allocates the receiver. */
  protected void allocate() {
    isAllocated = true;
    values.ensureCapacity(k);
  }

  /** Clears the receiver. */
  @Override
  public void clear() {
    values.clear();
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    DoubleBuffer copy = (DoubleBuffer) super.clone();
    if (this.values != null) {
      copy.values = copy.values.copy();
    }
    return copy;
  }

  /** Returns whether the specified element is contained in the receiver. */
  public boolean contains(double element) {
    this.sort();
    return values.contains(element);
  }

  /** Returns whether the receiver is empty. */
  @Override
  public boolean isEmpty() {
    return values.isEmpty();
  }

  /** Returns whether the receiver is empty. */
  @Override
  public boolean isFull() {
    return values.size() == k;
  }

  /**
   * Returns the number of elements currently needed to store all contained elements. This number usually differs from
   * the results of method <tt>size()</tt>, according to the underlying algorithm.
   */
  public int memory() {
    return values.elements().length;
  }

  /**
   * Returns the rank of a given element within the sorted sequence of the receiver. A rank is the number of elements <=
   * element. Ranks are of the form {1,2,...size()}. If no element is <= element, then the rank is zero. If the element
   * lies in between two contained elements, then uses linear interpolation.
   *
   * @param element the element to search for
   * @return the rank of the element.
   */
  public double rank(double element) {
    this.sort();
    return org.apache.mahout.math.jet.stat.Descriptive.rankInterpolated(this.values, element);
  }

  /** Returns the number of elements contained in the receiver. */
  @Override
  public int size() {
    return values.size();
  }

  /** Sorts the receiver. */
  @Override
  public void sort() {
    if (!this.isSorted) {
      // IMPORTANT: TO DO : replace mergeSort with quickSort!
      // currently it is mergeSort only for debugging purposes (JDK 1.2 can't be imported into VisualAge).
      values.sort();
      //values.mergeSort();
      this.isSorted = true;
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return "k=" + this.k +
        ", w=" + Long.toString(weight()) +
        ", l=" + Integer.toString(level()) +
        ", size=" + values.size();
    //", v=" + values.toString();
  }
}
