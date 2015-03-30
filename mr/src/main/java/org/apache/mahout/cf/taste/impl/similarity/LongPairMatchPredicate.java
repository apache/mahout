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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.common.LongPair;

/**
 * A {@link Cache.MatchPredicate} which will match an ID against either element of a
 * {@link LongPair}.
 */
final class LongPairMatchPredicate implements Cache.MatchPredicate<LongPair> {

  private final long id;

  LongPairMatchPredicate(long id) {
    this.id = id;
  }

  @Override
  public boolean matches(LongPair pair) {
    return pair.getFirst() == id || pair.getSecond() == id;
  }

}
