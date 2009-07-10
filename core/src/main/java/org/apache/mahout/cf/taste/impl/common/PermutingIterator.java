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

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

/** An {@link Iterator} that iterates in a random order over a given sequence of elements. It is non-destructive. */
public final class PermutingIterator<T> implements Iterator<T> {

  private final T[] elements;
  private final int[] permutation;
  private int offset;

  public PermutingIterator(T[] elements) {
    this.elements = elements;
    this.permutation = new int[elements.length];
    offset = 0;
    buildPermutation();
  }

  private void buildPermutation() {
    int length = permutation.length;
    for (int i = 0; i < length; i++) {
      permutation[i] = i;
    }
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < length - 1; i++) {
      int swapWith = i + r.nextInt(length - i);
      if (i != swapWith) {
        int temp = permutation[i];
        permutation[i] = permutation[swapWith];
        permutation[swapWith] = temp;
      }
    }
  }

  @Override
  public boolean hasNext() {
    return offset < elements.length;
  }

  @Override
  public T next() {
    if (offset >= elements.length) {
      throw new NoSuchElementException();
    }
    return elements[permutation[offset++]];
  }

  /** @throws UnsupportedOperationException always */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

}
