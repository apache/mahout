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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import java.io.IOException;

/**
 * Provides storage for {@link Factorization}s
 */
public interface PersistenceStrategy {

  /**
   * Load a factorization from a persistent store.
   *
   * @return a Factorization or null if the persistent store is empty.
   *
   * @throws IOException
   */
  Factorization load() throws IOException;

  /**
   * Write a factorization to a persistent store unless it already
   * contains an identical factorization.
   *
   * @param factorization
   *
   * @throws IOException
   */
  void maybePersist(Factorization factorization) throws IOException;

}
