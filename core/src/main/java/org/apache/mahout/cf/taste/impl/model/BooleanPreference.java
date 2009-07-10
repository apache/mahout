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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;

/**
 * Encapsulates a simple boolean "preference" for an {@link Item} whose value does not matter (is fixed at 1.0). This is
 * appropriate in situations where users conceptually have only a general "yes" preference for items, rather than a
 * spectrum of preference values.
 */
public final class BooleanPreference implements SettableUserPreference, Serializable {

  private User user;
  private final Item item;

  public BooleanPreference(User user, Item item) {
    if (item == null) {
      throw new IllegalArgumentException("item is null");
    }
    this.user = user;
    this.item = item;
  }

  @Override
  public User getUser() {
    if (user == null) {
      throw new IllegalStateException("User was never set");
    }
    return user;
  }

  @Override
  public void setUser(User user) {
    if (user == null) {
      throw new IllegalArgumentException("user is null");
    }
    this.user = user;
  }

  @Override
  public Item getItem() {
    return item;
  }

  @Override
  public double getValue() {
    return 1.0;
  }

  @Override
  public void setValue(double value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return "BooleanPreference[user: " + user + ", item:" + item + ']';
  }

}