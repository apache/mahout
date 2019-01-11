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

package org.apache.mahout.cf.taste.recommender;

/**
 * <p>
 * A {@link Rescorer} which operates on {@code long} primitive IDs, rather than arbitrary {@link Object}s.
 * This is provided since most uses of this interface in the framework take IDs (as {@code long}) as an
 * argument, and so this can be used to avoid unnecessary boxing/unboxing.
 * </p>
 */
public interface IDRescorer {
  
  /**
   * @param id
   *          ID of thing (user, item, etc.) to rescore
   * @param originalScore
   *          original score
   * @return modified score, or {@link Double#NaN} to indicate that this should be excluded entirely
   */
  double rescore(long id, double originalScore);
  
  /**
   * Returns {@code true} to exclude the given thing.
   *
   * @param id
   *          ID of thing (user, item, etc.) to rescore
   * @return {@code true} to exclude, {@code false} otherwise
   */
  boolean isFiltered(long id);
  
}
