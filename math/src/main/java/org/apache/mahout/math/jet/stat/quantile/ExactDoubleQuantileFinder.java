/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.function.DoubleProcedure;
import org.apache.mahout.math.list.DoubleArrayList;

/**
 * Exact quantile finding algorithm for known and unknown <tt>N</tt> requiring large main memory; computes quantiles
 * over a sequence of <tt>double</tt> elements. The folkore algorithm: Keeps all elements in main memory, sorts the
 * list, then picks the quantiles.
 */
//class ExactDoubleQuantileFinder extends Object implements DoubleQuantileFinder {
class ExactDoubleQuantileFinder extends PersistentObject implements DoubleQuantileFinder {

  private DoubleArrayList buffer;
  private boolean isSorted;

  /** Constructs an empty exact quantile finder. */
  ExactDoubleQuantileFinder() {
    this.buffer = new DoubleArrayList(0);
    this.clear();
  }

  /**
   * Adds a value to the receiver.
   *
   * @param value the value to add.
   */
  public void add(double value) {
    this.buffer.add(value);
    this.isSorted = false;
  }

  /**
   * Adds all values of the specified list to the receiver.
   *
   * @param values the list of which all values shall be added.
   */
  public void addAllOf(DoubleArrayList values) {
    addAllOfFromTo(values, 0, values.size() - 1);
  }

  /**
   * Adds the part of the specified list between indexes <tt>from</tt> (inclusive) and <tt>to</tt> (inclusive) to the
   * receiver.
   *
   * @param values the list of which elements shall be added.
   * @param from   the index of the first element to be added (inclusive).
   * @param to     the index of the last element to be added (inclusive).
   */
  public void addAllOfFromTo(DoubleArrayList values, int from, int to) {
    buffer.addAllOfFromTo(values, from, to);
    this.isSorted = false;
  }

  /**
   * Removes all elements from the receiver.  The receiver will be empty after this call returns, and its memory
   * requirements will be close to zero.
   */
  public void clear() {
    this.buffer.clear();
    this.buffer.trimToSize();
    this.isSorted = false;
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    ExactDoubleQuantileFinder copy = (ExactDoubleQuantileFinder) super.clone();
    if (this.buffer != null) {
      copy.buffer = copy.buffer.copy();
    }
    return copy;
  }

  /** Returns whether the specified element is contained in the receiver. */
  public boolean contains(double element) {
    this.sort();
    return buffer.binarySearch(element) >= 0;
  }

  /**
   * Applies a procedure to each element of the receiver, if any. Iterates over the receiver in no particular order.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  public boolean forEach(DoubleProcedure procedure) {
    double[] theElements = buffer.elements();
    int theSize = (int) size();

    for (int i = 0; i < theSize;) {
      if (!procedure.apply(theElements[i++])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the number of elements currently needed to store all contained elements. This number usually differs from
   * the results of method <tt>size()</tt>, according to the underlying datastructure.
   */
  public long memory() {
    return buffer.elements().length;
  }

  /**
   * Returns how many percent of the elements contained in the receiver are <tt>&lt;= element</tt>. Does linear
   * interpolation if the element is not contained but lies in between two contained elements.
   *
   * @param element the element to search for.
   * @return the percentage <tt>p</tt> of elements <tt>&lt;= element</tt> (<tt>0.0 &lt;= p &lt;=1.0)</tt>.
   */
  public double phi(double element) {
    this.sort();
    return org.apache.mahout.math.jet.stat.Descriptive.rankInterpolated(buffer, element) / this.size();
  }

  /**
   * Computes the specified quantile elements over the values previously added.
   *
   * @param phis the quantiles for which elements are to be computed. Each phi must be in the interval [0.0,1.0].
   *             <tt>phis</tt> must be sorted ascending.
   * @return the exact quantile elements.
   */
  public DoubleArrayList quantileElements(DoubleArrayList phis) {
    this.sort();
    return org.apache.mahout.math.jet.stat.Descriptive.quantiles(this.buffer, phis);
  }

  /**
   * Returns the number of elements currently contained in the receiver (identical to the number of values added so
   * far).
   */
  public long size() {
    return buffer.size();
  }

  /** Sorts the receiver. */
  protected void sort() {
    if (!isSorted) {
      // IMPORTANT: TO DO : replace mergeSort with quickSort!
      // currently it is mergeSort because JDK 1.2 can't be imported into VisualAge.
      buffer.sort();
      //this.buffer.mergeSort();
      this.isSorted = true;
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    String s = this.getClass().getName();
    s = s.substring(s.lastIndexOf('.') + 1);
    return s + "(mem=" + memory() + ", size=" + size() + ')';
  }

}
