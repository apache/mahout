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

package org.apache.mahout.common.iterator;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Handy source of well characterized Iterators and Iterables.
 */
final class Integers {

  private Integers() {
  }

  static Iterator<Integer> iterator(int n) {
    return new IntegerIterator(n);
  }

  static Iterable<Integer> iterable(final int n) {
    return new Iterable<Integer>() {
      @Override
      public Iterator<Integer> iterator() {
        return Integers.iterator(n);
      }
    };
  }

  private static class IntegerIterator implements Iterator<Integer> {

    private int v;
    private final int max;

    IntegerIterator(int n) {
      v = 0;
      max = n;
    }

    @Override
    public boolean hasNext() {
      return v < max;
    }

    @Override
    public Integer next() {
      if (v >= max) {
        throw new NoSuchElementException();
      }
      return v++;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException("Can't remove anything from the set of integers");
    }
  }
}
