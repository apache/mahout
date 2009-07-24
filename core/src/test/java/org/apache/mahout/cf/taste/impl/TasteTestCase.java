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

package org.apache.mahout.cf.taste.impl;

import junit.framework.TestCase;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class TasteTestCase extends TestCase {

  /** "Close enough" value for floating-point comparisons. */
  public static final double EPSILON = 0.00001;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
  }

  public static User getUser(String userID, Double... values) {
    List<Preference> prefs = new ArrayList<Preference>(values.length);
    int i = 0;
    for (Double value : values) {
      if (value != null) {
        prefs.add(new GenericPreference(null, String.valueOf(i), value));
      }
      i++;
    }
    return new GenericUser(userID, prefs);
  }

  public static DataModel getDataModel(User... users) {
    return new GenericDataModel(Arrays.asList(users));
  }

  public static DataModel getDataModel() {
    return new GenericDataModel(getMockUsers());
  }

  public static List<User> getMockUsers() {
    List<User> users = new ArrayList<User>(4);
    users.add(getUser("test1", 0.1, 0.3));
    users.add(getUser("test2", 0.2, 0.3, 0.3));
    users.add(getUser("test3", 0.4, 0.3, 0.5));
    users.add(getUser("test4", 0.7, 0.3, 0.8));
    return users;
  }

}
