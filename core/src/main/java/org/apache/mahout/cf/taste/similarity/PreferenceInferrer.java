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

package org.apache.mahout.cf.taste.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>Implementations of this interface compute an inferred preference for a user and an item that the
 * user has not expressed any preference for. This might be an average of other preferences scores from that user, for
 * example. This technique is sometimes called "default voting".</p>
 */
public interface PreferenceInferrer extends Refreshable {

  /**
   * <p>Infers the given user's preference value for an item.</p>
   *
   * @param userID ID of user to infer preference for
   * @param itemID item ID to infer preference for
   * @return inferred preference
   * @throws TasteException if an error occurs while inferring
   */
  float inferPreference(long userID, long itemID) throws TasteException;

}
