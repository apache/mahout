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

import java.util.NoSuchElementException;

import com.google.common.base.Preconditions;

/**
 * While long[] is an Iterable, it is not an Iterable&lt;Long&gt;. This adapter class addresses that.
 */
public final class LongPrimitiveArrayIterator implements LongPrimitiveIterator {
  
  private final long[] array;
  private int position;
  private final int max;
  
  /**
   * <p>
   * Creates an {@link LongPrimitiveArrayIterator} over an entire array.
   * </p>
   * 
   * @param array
   *          array to iterate over
   */
  public LongPrimitiveArrayIterator(long[] array) {
    this.array = Preconditions.checkNotNull(array); // yeah, not going to copy the array here, for performance
    this.position = 0;
    this.max = array.length;
  }
  
  @Override
  public boolean hasNext() {
    return position < max;
  }
  
  @Override
  public Long next() {
    return nextLong();
  }
  
  @Override
  public long nextLong() {
    if (position >= array.length) {
      throw new NoSuchElementException();
    }
    return array[position++];
  }
  
  @Override
  public long peek() {
    if (position >= array.length) {
      throw new NoSuchElementException();
    }
    return array[position];
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
  public String toString() {
    return "LongPrimitiveArrayIterator";
  }
  
}
