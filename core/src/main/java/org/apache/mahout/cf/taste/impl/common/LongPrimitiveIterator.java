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
 * Adds notion of iterating over <code>long</code> primitives in the style of an
 * {@link Iterator} -- as opposed to iterating over {@link Long}. Implementations of
 * this interface however also implement {@link Iterator} and {@link Iterable} over
 * {@link Long} for convenience.
 */
public interface LongPrimitiveIterator extends Iterator<Long> {

  /**
   * @return next <code>long</code> in iteration
   * @throws java.util.NoSuchElementException if no more elements exist in the iteration
   */
  long nextLong();

  /**
   * @return next <code>long</code> in iteration without advancing iteration
   */
  long peek();

}