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

import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;

/** <p>A simple {@link Preference} encapsulating an item and preference value.</p> */
public class GenericPreference implements SettableUserPreference, Serializable {

  private User user;
  private final Comparable<?> itemID;
  private double value;

  public GenericPreference(User user, Comparable<?> itemID, double value) {
    if (itemID == null) {
      throw new IllegalArgumentException("itemID is null");
    }
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }
    this.user = user;
    this.itemID = itemID;
    this.value = value;
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
  public Comparable<?> getItemID() {
    return itemID;
  }

  @Override
  public double getValue() {
    return value;
  }

  @Override
  public void setValue(double value) {
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }
    this.value = value;
  }

  @Override
  public String toString() {
    return "GenericPreference[user: " + user + ", itemID:" + itemID + ", value:" + value + ']';
  }

}
