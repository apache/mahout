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

package org.apache.mahout.cf.taste.common;

import java.util.Collection;

/**
 * <p>
 * Implementations of this interface have state that can be periodically refreshed. For example, an
 * implementation instance might contain some pre-computed information that should be periodically refreshed.
 * The {@link #refresh(Collection)} method triggers such a refresh.
 * </p>
 * 
 * <p>
 * All Taste components implement this. In particular,
 * {@link org.apache.mahout.cf.taste.recommender.Recommender}s do. Callers may want to call
 * {@link #refresh(Collection)} periodically to re-compute information throughout the system and bring it up
 * to date, though this operation may be expensive.
 * </p>
 */
public interface Refreshable {
  
  /**
   * <p>
   * Triggers "refresh" -- whatever that means -- of the implementation. The general contract is that any
   * {@link Refreshable} should always leave itself in a consistent, operational state, and that the refresh
   * atomically updates internal state from old to new.
   * </p>
   * 
   * @param alreadyRefreshed
   *          {@link org.apache.mahout.cf.taste.common.Refreshable}s that are known to have already been
   *          refreshed as a result of an initial call to a {@link #refresh(Collection)} method on some
   *          object. This ensure that objects in a refresh dependency graph aren't refreshed twice
   *          needlessly.
   */
  void refresh(Collection<Refreshable> alreadyRefreshed);
  
}
