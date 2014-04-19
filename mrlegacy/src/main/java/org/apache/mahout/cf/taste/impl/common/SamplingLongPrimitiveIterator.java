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
import org.apache.commons.math3.distribution.PascalDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

/**
 * Wraps a {@link LongPrimitiveIterator} and returns only some subset of the elements that it would,
 * as determined by a sampling rate parameter.
 */
public final class SamplingLongPrimitiveIterator extends AbstractLongPrimitiveIterator {
  
  private final PascalDistribution geometricDistribution;
  private final LongPrimitiveIterator delegate;
  private long next;
  private boolean hasNext;
  
  public SamplingLongPrimitiveIterator(LongPrimitiveIterator delegate, double samplingRate) {
    this(RandomUtils.getRandom(), delegate, samplingRate);
  }

  public SamplingLongPrimitiveIterator(RandomWrapper random, LongPrimitiveIterator delegate, double samplingRate) {
    Preconditions.checkNotNull(delegate);
    Preconditions.checkArgument(samplingRate > 0.0 && samplingRate <= 1.0, "Must be: 0.0 < samplingRate <= 1.0");
    // Geometric distribution is special case of negative binomial (aka Pascal) with r=1:
    geometricDistribution = new PascalDistribution(random.getRandomGenerator(), 1, samplingRate);
    this.delegate = delegate;
    this.hasNext = true;
    doNext();
  }
  
  @Override
  public boolean hasNext() {
    return hasNext;
  }
  
  @Override
  public long nextLong() {
    if (hasNext) {
      long result = next;
      doNext();
      return result;
    }
    throw new NoSuchElementException();
  }
  
  @Override
  public long peek() {
    if (hasNext) {
      return next;
    }
    throw new NoSuchElementException();
  }
  
  private void doNext() {
    int toSkip = geometricDistribution.sample();
    delegate.skip(toSkip);
    if (delegate.hasNext()) {
      next = delegate.next();
    } else {
      hasNext = false;
    }
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
    int toSkip = 0;
    for (int i = 0; i < n; i++) {
      toSkip += geometricDistribution.sample();
    }
    delegate.skip(toSkip);
    if (delegate.hasNext()) {
      next = delegate.next();
    } else {
      hasNext = false;
    }
  }
  
  public static LongPrimitiveIterator maybeWrapIterator(LongPrimitiveIterator delegate, double samplingRate) {
    return samplingRate >= 1.0 ? delegate : new SamplingLongPrimitiveIterator(delegate, samplingRate);
  }
  
}
