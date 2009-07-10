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

package org.apache.mahout.cf.taste.impl.common;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

/** <p>Simple, fast {@link Iterator} for an array.</p> */
public final class ArrayIterator<T> implements SkippingIterator<T>, Iterable<T> {

  private final T[] array;
  private int position;
  private final int max;

  /**
   * <p>Creates an {@link ArrayIterator} over an entire array.</p>
   *
   * @param array array to iterate over
   */
  public ArrayIterator(T[] array) {
    if (array == null) {
      throw new IllegalArgumentException("array is null");
    }
    this.array = array; // yeah, not going to copy the array here, for performance
    this.position = 0;
    this.max = array.length;
  }

  @Override
  public boolean hasNext() {
    return position < max;
  }

  @Override
  public T next() {
    if (position >= array.length) {
      throw new NoSuchElementException();
    }
    return array[position++];
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void skip(int n) {
    if (n > 0) {
      position += n;
    }
  }

  @Override
  public Iterator<T> iterator() {
    return this;
  }

  @Override
  public String toString() {
    return "ArrayIterator[" + Arrays.toString(array) + '@' + position + ']';
  }

}
