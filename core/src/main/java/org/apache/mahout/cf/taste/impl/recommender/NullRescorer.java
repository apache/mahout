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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.impl.common.LongPair;
import org.apache.mahout.cf.taste.recommender.Rescorer;

/** <p>A simple {@link Rescorer} which always returns the original score.</p> */
public final class NullRescorer<T> implements Rescorer<T> {

  private static final Rescorer<Long> userOrItemInstance = new NullRescorer<Long>();
  private static final Rescorer<LongPair> itemItemPairInstance = new NullRescorer<LongPair>();
  private static final Rescorer<LongPair> userUserPairInstance = new NullRescorer<LongPair>();

  public static Rescorer<Long> getItemInstance() {
    return userOrItemInstance;
  }

  public static Rescorer<Long> getUserInstance() {
    return userOrItemInstance;
  }

  public static Rescorer<LongPair> getItemItemPairInstance() {
    return itemItemPairInstance;
  }

  public static Rescorer<LongPair> getUserUserPairInstance() {
    return userUserPairInstance;
  }

  private NullRescorer() {
    // do nothing
  }

  /**
   * @param thing         to rescore
   * @param originalScore current score for item
   * @return same originalScore as new score, always
   */
  @Override
  public double rescore(T thing, double originalScore) {
    return originalScore;
  }

  @Override
  public boolean isFiltered(T thing) {
    return false;
  }

  @Override
  public String toString() {
    return "NullRescorer";
  }

}
