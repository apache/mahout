/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.list;

import org.apache.mahout.matrix.PersistentObject;

import java.util.List;

/**
 Abstract base class for resizable collections holding objects or primitive data types such as <code>int</code>, <code>float</code>, etc.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Note that this implementation is not synchronized.</b>

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 @see     ArrayList
 */
public abstract class AbstractCollection<T> extends PersistentObject {

  /** Removes all elements from the receiver.  The receiver will be empty after this call returns. */
  public abstract void clear();

  /**
   * Tests if the receiver has no elements.
   *
   * @return <code>true</code> if the receiver has no elements; <code>false</code> otherwise.
   */
  public boolean isEmpty() {
    return size() == 0;
  }

  /** Returns the number of elements contained in the receiver. */
  public abstract int size();

  /** Returns a <code>List</code> containing all the elements in the receiver. */
  public abstract List<T> toList();

  /** Returns a string representation of the receiver, containing the String representation of each element. */
  public String toString() {
    return toList().toString();
  }
}
