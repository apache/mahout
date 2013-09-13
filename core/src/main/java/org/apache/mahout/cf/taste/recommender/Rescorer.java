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
 * A {@link Rescorer} simply assigns a new "score" to a thing like an ID of an item or user which a
 * {@link Recommender} is considering returning as a top recommendation. It may be used to arbitrarily re-rank
 * the results according to application-specific logic before returning recommendations. For example, an
 * application may want to boost the score of items in a certain category just for one request.
 * </p>
 *
 * <p>
 * A {@link Rescorer} can also exclude a thing from consideration entirely by returning {@code true} from
 * {@link #isFiltered(Object)}.
 * </p>
 */
public interface Rescorer<T> {
  
  /**
   * @param thing
   *          thing to rescore
   * @param originalScore
   *          original score
   * @return modified score, or {@link Double#NaN} to indicate that this should be excluded entirely
   */
  double rescore(T thing, double originalScore);
  
  /**
   * Returns {@code true} to exclude the given thing.
   *
   * @param thing
   *          the thing to filter
   * @return {@code true} to exclude, {@code false} otherwise
   */
  boolean isFiltered(T thing);
}
