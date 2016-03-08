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
 * Adds ability to skip ahead in an iterator, perhaps more efficiently than by calling {@link #next()}
 * repeatedly.
 */
public interface SkippingIterator<V> extends Iterator<V> {
  
  /**
   * Skip the next n elements supplied by this {@link Iterator}. If there are less than n elements remaining,
   * this skips all remaining elements in the {@link Iterator}. This method has the same effect as calling
   * {@link #next()} n times, except that it will never throw {@link java.util.NoSuchElementException}.
   */
  void skip(int n);
  
}
