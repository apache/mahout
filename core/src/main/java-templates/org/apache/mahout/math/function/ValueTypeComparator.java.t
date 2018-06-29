/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 
package org.apache.mahout.math.function;

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/

/**
 * A comparison function which imposes a <i>total ordering</i> on some collection of elements.  Comparators can be
 * passed to a sort method (such as <tt>org.apache.mahout.math.Sorting.quickSort</tt>) to allow precise control over
 * the sort order.<p>
 *
 * Note: It is generally a good idea for comparators to implement <tt>java.io.Serializable</tt>, as they may be used as
 * ordering methods in serializable data structures.  In order for the data structure to serialize successfully, the
 * comparator (if provided) must implement <tt>Serializable</tt>.<p>
 *
 * @see java.util.Comparator
 * @see org.apache.mahout.math.Sorting
 */
public interface ${valueTypeCap}Comparator {

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
  int compare(${valueType} o1, ${valueType} o2);

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
   * @see Object#hashCode()
   */
  boolean equals(Object obj);
}
