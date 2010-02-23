/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.matrix.DoubleMatrix1D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface DoubleMatrix1DComparator {

  /**
   * Compares its two arguments for order.  Returns a negative integer, zero, or a positive integer as the first
   * argument is less than, equal to, or greater than the second.<p>
   *
   * The implementor must ensure that <tt>sgn(compare(x, y)) == -sgn(compare(y, x))</tt> for all <tt>x</tt> and
   * <tt>y</tt>.  (This implies that <tt>compare(x, y)</tt> must throw an exception if and only if <tt>compare(y,
   * x)</tt> throws an exception.)<p>
   *
   * The implementor must also ensure that the relation is transitive: <tt>((compare(x, y)&gt;0) &amp;&amp; (compare(y,
   * z)&gt;0))</tt> implies <tt>compare(x, z)&gt;0</tt>.<p>
   *
   * Finally, the implementer must ensure that <tt>compare(x, y)==0</tt> implies that <tt>sgn(compare(x,
   * z))==sgn(compare(y, z))</tt> for all <tt>z</tt>.<p>
   *
   * @return a negative integer, zero, or a positive integer as the first argument is less than, equal to, or greater
   *         than the second.
   */
  int compare(DoubleMatrix1D o1, DoubleMatrix1D o2);

  /**
   * Indicates whether some other object is &quot;equal to&quot; this Comparator.  This method must obey the general
   * contract of <tt>Object.equals(Object)</tt>.  Additionally, this method can return <tt>true</tt> <i>only</i> if the
   * specified Object is also a comparator and it imposes the same ordering as this comparator.  Thus,
   * <code>comp1.equals(comp2)</code> implies that <tt>sgn(comp1.compare(o1, o2))==sgn(comp2.compare(o1, o2))</tt> for
   * every element <tt>o1</tt> and <tt>o2</tt>.<p>
   *
   * Note that it is <i>always</i> safe <i>not</i> to override <tt>Object.equals(Object)</tt>.  However, overriding this
   * method may, in some cases, improve performance by allowing programs to determine that two distinct Comparators
   * impose the same order.
   *
   * @param obj the reference object with which to compare.
   * @return <code>true</code> only if the specified object is also a comparator and it imposes the same ordering as
   *         this comparator.
   * @see java.lang.Object#hashCode()
   */
  boolean equals(Object obj);
}
