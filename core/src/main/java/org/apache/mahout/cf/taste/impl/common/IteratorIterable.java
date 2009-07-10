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

/**
 * <p>Simple utility class that makes an {@link Iterator} {@link Iterable} by returning the {@link Iterator}
 * itself.</p>
 */
public final class IteratorIterable<T> implements Iterable<T> {

  private Iterator<T> iterator;

  /**
   * <p>Constructs an {@link IteratorIterable} for an {@link Iterator}.</p>
   *
   * @param iterator {@link Iterator} on which to base this {@link IteratorIterable}
   */
  public IteratorIterable(Iterator<T> iterator) {
    if (iterator == null) {
      throw new IllegalArgumentException("iterator is null");
    }
    this.iterator = iterator;
  }

  @Override
  public Iterator<T> iterator() {
    if (iterator == null) {
      throw new IllegalStateException("iterator() has already been called");
    }
    Iterator<T> result = iterator;
    iterator = null;
    return result;
  }

  @Override
  public String toString() {
    return "IteratorIterable[iterator:" + iterator + ']';
  }

}
