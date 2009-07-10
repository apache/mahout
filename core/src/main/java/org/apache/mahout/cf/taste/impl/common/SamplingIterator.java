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

/**
 * Wraps an {@link Iterator} and returns only some subset of the elements that it would, as determined by a sampling
 * rate parameter.
 */
public final class SamplingIterator<T> implements Iterator<T> {

  private static final Random r = RandomUtils.getRandom();

  private final Iterator<? extends T> delegate;
  private final double samplingRate;
  private T next;
  private boolean hasNext;

  public SamplingIterator(Iterator<? extends T> delegate, double samplingRate) {
    this.delegate = delegate;
    this.samplingRate = samplingRate;
    this.hasNext = true;
    doNext();
  }

  @Override
  public boolean hasNext() {
    return hasNext;
  }

  @Override
  public T next() {
    if (hasNext) {
      T result = next;
      doNext();
      return result;
    }
    throw new NoSuchElementException();
  }

  private void doNext() {
    boolean found = false;
    if (delegate instanceof SkippingIterator) {
      SkippingIterator<? extends T> skippingDelegate = (SkippingIterator<? extends T>) delegate;
      int toSkip = 0;
      while (r.nextDouble() >= samplingRate) {
        toSkip++;
      }
      // Really, would be nicer to select value from geometric distribution, for small values of samplingRate
      if (toSkip > 0) {
        skippingDelegate.skip(toSkip);
      }
      if (skippingDelegate.hasNext()) {
        next = skippingDelegate.next();
        found = true;
      }
    } else {
      while (delegate.hasNext()) {
        T delegateNext = delegate.next();
        if (r.nextDouble() < samplingRate) {
          next = delegateNext;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      hasNext = false;
      next = null;
    }
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

}
