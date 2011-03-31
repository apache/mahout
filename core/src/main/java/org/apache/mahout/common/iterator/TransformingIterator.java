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

/**
 * An iterator that delegates to another iterator but transforms its values.
 */
public abstract class TransformingIterator<I,O> implements Iterator<O> {

  private final Iterator<? extends I> delegate;

  protected TransformingIterator(Iterator<? extends I> delegate) {
    this.delegate = delegate;
  }

  /**
   * @param in underlying iterator's value
   * @return the transformed value returned from this iterator
   */
  protected abstract O transform(I in);
  
  @Override
  public final boolean hasNext() {
    return delegate.hasNext();
  }
  
  @Override
  public final O next() {
    return transform(delegate.next());
  }
  
  @Override
  public final void remove() {
    delegate.remove();
  }
  
}
