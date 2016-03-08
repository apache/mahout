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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.impl.common.AbstractLongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

final class PlusAnonymousUserLongPrimitiveIterator extends AbstractLongPrimitiveIterator {
  
  private final LongPrimitiveIterator delegate;
  private final long extraDatum;
  private boolean datumConsumed;
  
  PlusAnonymousUserLongPrimitiveIterator(LongPrimitiveIterator delegate, long extraDatum) {
    this.delegate = delegate;
    this.extraDatum = extraDatum;
    datumConsumed = false;
  }
  
  @Override
  public long nextLong() {
    if (datumConsumed) {
      return delegate.nextLong();
    } else {
      if (delegate.hasNext()) {
        long delegateNext = delegate.peek();
        if (extraDatum <= delegateNext) {
          datumConsumed = true;
          return extraDatum;
        } else {
          return delegate.next();
        }
      } else {
        datumConsumed = true;
        return extraDatum;
      }
    }
  }
  
  @Override
  public long peek() {
    if (datumConsumed) {
      return delegate.peek();
    } else {
      if (delegate.hasNext()) {
        long delegateNext = delegate.peek();
        if (extraDatum <= delegateNext) {
          return extraDatum;
        } else {
          return delegateNext;
        }
      } else {
        return extraDatum;
      }
    }
  }
  
  @Override
  public boolean hasNext() {
    return !datumConsumed || delegate.hasNext();
  }
  
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void skip(int n) {
    for (int i = 0; i < n; i++) {
      nextLong();
    }
  }
  
}
