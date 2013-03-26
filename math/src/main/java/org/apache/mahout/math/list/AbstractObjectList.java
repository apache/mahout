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
/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.list;

import java.util.Collection;

/**
 Abstract base class for resizable lists holding objects or primitive data types such as <code>int</code>,
 <code>float</code>, etc.First see the <a href="package-summary.html">package summary</a> and
 javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Note that this implementation is not synchronized.</b>

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 @see      java.util.ArrayList
 @see      java.util.Vector
 @see      java.util.Arrays
 */
public abstract class AbstractObjectList<T> extends AbstractList {

  /**
   * Appends all of the elements of the specified Collection to the receiver.
   *
   * @throws ClassCastException if an element in the collection is not of the same parameter type of the receiver.
   */
  public void addAllOf(Collection<T> collection) {
    this.beforeInsertAllOf(size(), collection);
  }

  /**
   * Inserts all elements of the specified collection before the specified position into the receiver. Shifts the
   * element currently at that position (if any) and any subsequent elements to the right (increases their indices).
   *
   * @param index      index before which to insert first element from the specified collection.
   * @param collection the collection to be inserted
   * @throws ClassCastException        if an element in the collection is not of the same parameter type of the
   *                                   receiver.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt; size()</tt>.
   */
  public void beforeInsertAllOf(int index, Collection<T> collection) {
    this.beforeInsertDummies(index, collection.size());
    this.replaceFromWith(index, collection);
  }

  /**
   * Replaces the part of the receiver starting at <code>from</code> (inclusive) with all the elements of the specified
   * collection. Does not alter the size of the receiver. Replaces exactly <tt>Math.max(0,Math.min(size()-from,
   * other.size()))</tt> elements.
   *
   * @param from  the index at which to copy the first element from the specified collection.
   * @param other Collection to replace part of the receiver
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt;= size()</tt>.
   */
  public abstract void replaceFromWith(int from, Collection<T> other);
}
