/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

import org.apache.mahout.math.function.DoubleProcedure;
import org.apache.mahout.math.list.DoubleArrayList;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface DoubleQuantileFinder extends java.io.Serializable {
  //public interface DoubleQuantileFinder extends com.objy.db.iapp.PersistentEvents, java.io.Serializable {

  /**
   * Adds a value to the receiver.
   *
   * @param value the value to add.
   */
  void add(double value);

  /**
   * Adds all values of the specified list to the receiver.
   *
   * @param values the list of which all values shall be added.
   */
  void addAllOf(DoubleArrayList values);

  /**
   * Adds the part of the specified list between indexes <tt>from</tt> (inclusive) and <tt>to</tt> (inclusive) to the
   * receiver.
   *
   * @param values the list of which elements shall be added.
   * @param from   the index of the first element to be added (inclusive).
   * @param to     the index of the last element to be added (inclusive).
   */
  void addAllOfFromTo(DoubleArrayList values, int from, int to);

  /**
   * Removes all elements from the receiver.  The receiver will be empty after this call returns, and its memory
   * requirements will be close to zero.
   */
  void clear();

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  Object clone();

  /**
   * Applies a procedure to each element of the receiver, if any. Iterates over the receiver in no particular order.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  boolean forEach(DoubleProcedure procedure);

  /**
   * Returns the number of elements currently needed to store all contained elements. This number usually differs from
   * the results of method <tt>size()</tt>, according to the underlying datastructure.
   */
  long memory();

  /**
   * Returns how many percent of the elements contained in the receiver are <tt>&lt;= element</tt>. Does linear
   * interpolation if the element is not contained but lies in between two contained elements.
   *
   * Writing a wrapper is a good idea if you can think of better ways of doing interpolation. Same if you want to keep
   * min,max and other such measures.
   *
   * @param element the element to search for.
   * @return the percentage <tt>p</tt> of elements <tt>&lt;= element</tt> (<tt>0.0 &lt;= p &lt;=1.0)</tt>.
   */
  double phi(double element);

  /**
   * Computes the specified quantile elements over the values previously added.
   *
   * @param phis the quantiles for which elements are to be computed. Each phi must be in the interval [0.0,1.0].
   *             <tt>phis</tt> must be sorted ascending.
   * @return the quantile elements.
   */
  DoubleArrayList quantileElements(DoubleArrayList phis);

  /**
   * Returns the number of elements currently contained in the receiver (identical to the number of values added so
   * far).
   */
  long size();

}
