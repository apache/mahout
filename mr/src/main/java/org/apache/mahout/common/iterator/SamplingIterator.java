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

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import org.apache.commons.math3.distribution.PascalDistribution;
import org.apache.mahout.cf.taste.impl.common.SkippingIterator;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

/**
 * Wraps an {@link Iterator} and returns only some subset of the elements that it would, as determined by a
 * iterator rate parameter.
 */
public final class SamplingIterator<T> extends AbstractIterator<T> {
  
  private final PascalDistribution geometricDistribution;
  private final Iterator<? extends T> delegate;

  public SamplingIterator(Iterator<? extends T> delegate, double samplingRate) {
    this(RandomUtils.getRandom(), delegate, samplingRate);
  }

  public SamplingIterator(RandomWrapper random, Iterator<? extends T> delegate, double samplingRate) {
    Preconditions.checkNotNull(delegate);
    Preconditions.checkArgument(samplingRate > 0.0 && samplingRate <= 1.0,
        "Must be: 0.0 < samplingRate <= 1.0. But samplingRate = " + samplingRate);
    // Geometric distribution is special case of negative binomial (aka Pascal) with r=1:
    geometricDistribution = new PascalDistribution(random.getRandomGenerator(), 1, samplingRate);
    this.delegate = delegate;
  }

  @Override
  protected T computeNext() {
    int toSkip = geometricDistribution.sample();
    if (delegate instanceof SkippingIterator<?>) {
      SkippingIterator<? extends T> skippingDelegate = (SkippingIterator<? extends T>) delegate;
      skippingDelegate.skip(toSkip);
      if (skippingDelegate.hasNext()) {
        return skippingDelegate.next();
      }
    } else {
      for (int i = 0; i < toSkip && delegate.hasNext(); i++) {
        delegate.next();
      }
      if (delegate.hasNext()) {
        return delegate.next();
      }
    }
    return endOfData();
  }


  
}
