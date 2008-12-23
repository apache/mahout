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

import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.Rescorer;

/**
 * <p>A simple {@link Rescorer} which always returns the original score.</p>
 */
public final class NullRescorer<T> implements Rescorer<T> {

  private static final Rescorer<Item> itemInstance = new NullRescorer<Item>();
  private static final Rescorer<User> userInstance = new NullRescorer<User>();
  private static final Rescorer<Pair<Item, Item>> itemItemPairInstance = new NullRescorer<Pair<Item, Item>>();
  private static final Rescorer<Pair<User, User>> userUserPairInstance = new NullRescorer<Pair<User, User>>();

  public static Rescorer<Item> getItemInstance() {
    return itemInstance;
  }

  public static Rescorer<User> getUserInstance() {
    return userInstance;
  }

  public static Rescorer<Pair<Item, Item>> getItemItemPairInstance() {
    return itemItemPairInstance;
  }

  public static Rescorer<Pair<User, User>> getUserUserPairInstance() {
    return userUserPairInstance;
  }

  private NullRescorer() {
    // do nothing
  }

  /**
   * @param thing to rescore
   * @param originalScore current score for {@link Item}
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
