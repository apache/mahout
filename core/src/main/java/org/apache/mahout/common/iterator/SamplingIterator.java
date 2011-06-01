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
import java.util.Random;

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.cf.taste.impl.common.SkippingIterator;
import org.apache.mahout.common.RandomUtils;

/**
 * Wraps an {@link Iterator} and returns only some subset of the elements that it would, as determined by a
 * iterator rate parameter.
 */
public final class SamplingIterator<T> extends AbstractIterator<T> {
  
  private final Random random;
  private final Iterator<? extends T> delegate;
  private final double samplingRate;
  
  public SamplingIterator(Iterator<? extends T> delegate, double samplingRate) {
    random = RandomUtils.getRandom();
    this.delegate = delegate;
    this.samplingRate = samplingRate;
  }

  @Override
  protected T computeNext() {
    if (delegate instanceof SkippingIterator<?>) {
      SkippingIterator<? extends T> skippingDelegate = (SkippingIterator<? extends T>) delegate;
      int toSkip = 0;
      while (random.nextDouble() >= samplingRate) {
        toSkip++;
      }
      // Really, would be nicer to select value from geometric distribution, for small values of samplingRate
      if (toSkip > 0) {
        skippingDelegate.skip(toSkip);
      }
      if (skippingDelegate.hasNext()) {
        return skippingDelegate.next();
      }
    } else {
      while (delegate.hasNext()) {
        T delegateNext = delegate.next();
        if (random.nextDouble() < samplingRate) {
          return delegateNext;
        }
      }
    }
    return endOfData();
  }


  
}
