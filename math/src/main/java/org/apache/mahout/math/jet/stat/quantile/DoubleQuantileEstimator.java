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
 * The abstract base class for approximate quantile finders computing quantiles over a sequence of <tt>double</tt>
 * elements.
 */
//abstract class ApproximateDoubleQuantileFinder extends Object implements DoubleQuantileFinder {
abstract class DoubleQuantileEstimator extends PersistentObject
    implements DoubleQuantileFinder {

  protected DoubleBufferSet bufferSet;
  protected DoubleBuffer currentBufferToFill;
  protected int totalElementsFilled;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected DoubleQuantileEstimator() {
  }

  /**
   * Adds a value to the receiver.
   *
   * @param value the value to add.
   */
  @Override
  public void add(double value) {
    totalElementsFilled++;
    if (!sampleNextElement()) {
      return;
    }

    if (currentBufferToFill == null) {
      if (bufferSet._getFirstEmptyBuffer() == null) {
        collapse();
      }
      newBuffer();
    }

    currentBufferToFill.add(value);
    if (currentBufferToFill.isFull()) {
      currentBufferToFill = null;
    }
  }

  /**
   * Adds all values of the specified list to the receiver.
   *
   * @param values the list of which all values shall be added.
   */
  @Override
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
  @Override
  public void addAllOfFromTo(DoubleArrayList values, int from, int to) {

    double[] valuesToAdd = values.elements();
    int k = this.bufferSet.k();
    int bufferSize = k;
    double[] bufferValues = null;
    if (currentBufferToFill != null) {
      bufferValues = currentBufferToFill.values.elements();
      bufferSize = currentBufferToFill.size();
    }

    for (int i = from - 1; ++i <= to;) {
      if (sampleNextElement()) {
        if (bufferSize == k) { // full
          if (bufferSet._getFirstEmptyBuffer() == null) {
            collapse();
          }
          newBuffer();
          if (!currentBufferToFill.isAllocated) {
            currentBufferToFill.allocate();
          }
          currentBufferToFill.isSorted = false;
          bufferValues = currentBufferToFill.values.elements();
          bufferSize = 0;
        }

        bufferValues[bufferSize++] = valuesToAdd[i];
        if (bufferSize == k) { // full
          currentBufferToFill.values.setSize(bufferSize);
          currentBufferToFill = null;
        }
      }
    }
    if (this.currentBufferToFill != null) {
      this.currentBufferToFill.values.setSize(bufferSize);
    }

    this.totalElementsFilled += to - from + 1;
  }

  /** Not yet commented. */
  protected DoubleBuffer[] buffersToCollapse() {
    int minLevel = bufferSet._getMinLevelOfFullOrPartialBuffers();
    return bufferSet._getFullOrPartialBuffersWithLevel(minLevel);
  }

  /**
   * Removes all elements from the receiver.  The receiver will be empty after this call returns, and its memory
   * requirements will be close to zero.
   */
  @Override
  public void clear() {
    this.totalElementsFilled = 0;
    this.currentBufferToFill = null;
    this.bufferSet.clear();
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    DoubleQuantileEstimator copy = (DoubleQuantileEstimator) super.clone();
    if (this.bufferSet != null) {
      copy.bufferSet = (DoubleBufferSet) copy.bufferSet.clone();
      if (this.currentBufferToFill != null) {
        for (int index = 0; index < bufferSet.buffers.length; index ++) {
          if (bufferSet.buffers[index].equals(currentBufferToFill)) {
            copy.currentBufferToFill = copy.bufferSet.buffers[index];
            return copy;
          }
        }
        throw new IllegalStateException("current buffer not on the list.");
      }
    }
    return copy;
  }

  /** Not yet commented. */
  protected void collapse() {
    DoubleBuffer[] toCollapse = buffersToCollapse();
    DoubleBuffer outputBuffer = bufferSet.collapse(toCollapse);

    int minLevel = toCollapse[0].level();
    outputBuffer.level(minLevel + 1);

    postCollapse(toCollapse);
  }

  /** Returns whether the specified element is contained in the receiver. */
  public boolean contains(double element) {
    return bufferSet.contains(element);
  }

  /**
   * Applies a procedure to each element of the receiver, if any. Iterates over the receiver in no particular order.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  @Override
  public boolean forEach(DoubleProcedure procedure) {
    return this.bufferSet.forEach(procedure);
  }

  /**
   * Returns the number of elements currently needed to store all contained elements. This number usually differs from
   * the results of method <tt>size()</tt>, according to the underlying datastructure.
   */
  @Override
  public long memory() {
    return bufferSet.memory();
  }

  /** Not yet commented. */
  protected abstract void newBuffer();

  /**
   * Returns how many percent of the elements contained in the receiver are <tt>&lt;= element</tt>. Does linear
   * interpolation if the element is not contained but lies in between two contained elements.
   *
   * @param element the element to search for.
   * @return the percentage <tt>p</tt> of elements <tt>&lt;= element</tt> (<tt>0.0 &lt;= p &lt;=1.0)</tt>.
   */
  @Override
  public double phi(double element) {
    return bufferSet.phi(element);
  }

  /** Not yet commented. */
  protected abstract void postCollapse(DoubleBuffer[] toCollapse);

  /** Default implementation does nothing. */
  protected DoubleArrayList preProcessPhis(DoubleArrayList phis) {
    return phis;
  }

  /**
   * Computes the specified quantile elements over the values previously added.
   *
   * @param phis the quantiles for which elements are to be computed. Each phi must be in the interval [0.0,1.0].
   *             <tt>phis</tt> must be sorted ascending.
   * @return the approximate quantile elements.
   */
  @Override
  public DoubleArrayList quantileElements(DoubleArrayList phis) {

    phis = preProcessPhis(phis);

    long[] triggerPositions = new long[phis.size()];
    long totalSize = this.bufferSet.totalSize();
    for (int i = phis.size(); --i >= 0;) {
      triggerPositions[i] = Utils.epsilonCeiling(phis.get(i) * totalSize) - 1;
    }


    DoubleBuffer[] fullBuffers = bufferSet._getFullOrPartialBuffers();
    //double[] quantileElements = new double[phis.size()];

    //do the main work: determine values at given positions in sorted sequence
    return new DoubleArrayList(bufferSet.getValuesAtPositions(fullBuffers, triggerPositions));
  }

  /** Not yet commented. */
  protected abstract boolean sampleNextElement();

  /** Initializes the receiver */
  protected void setUp(int b, int k) {
    if (!(b >= 2 && k >= 1)) {
      throw new IllegalArgumentException("Assertion: b>=2 && k>=1");
    }
    this.bufferSet = new DoubleBufferSet(b, k);
    this.clear();
  }

  /**
   * Returns the number of elements currently contained in the receiver (identical to the number of values added so
   * far).
   */
  @Override
  public long size() {
    return totalElementsFilled;
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    String s = this.getClass().getName();
    s = s.substring(s.lastIndexOf('.') + 1);
    int b = bufferSet.b();
    int k = bufferSet.k();
    return s + "(mem=" + memory() + ", b=" + b + ", k=" + k + ", size=" + size() + ", totalSize=" +
        this.bufferSet.totalSize() + ')';
  }

}
