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

package org.apache.mahout.cf.taste.transforms;

import org.apache.mahout.cf.taste.common.Refreshable;

/**
 * <p>Implementations encapsulate some transformation on similarity values between two things, where things might be
 * IDs of users or items or something else.</p>
 */
public interface SimilarityTransform extends Refreshable {

  /**
   * @param value original similarity between thing1 and thing2 (should be in [-1,1])
   * @return transformed similarity (should be in [-1,1])
   */
  double transformSimilarity(long id1, long id2, double value);

}
