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

package org.apache.mahout.cf.taste.example.bookcrossing;

import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

import java.util.Collection;

public final class GeoUserSimilarity implements UserSimilarity {

  private final BookCrossingDataModel model;

  public GeoUserSimilarity(BookCrossingDataModel model) {
    this.model = model;
  }

  @Override
  public double userSimilarity(User user1, User user2) throws TasteException {
    BookCrossingUser bcUser1 = user1 instanceof BookCrossingUser ? (BookCrossingUser) user1 : (BookCrossingUser) model.getUser(user1.getID());
    BookCrossingUser bcUser2 = user2 instanceof BookCrossingUser ? (BookCrossingUser) user2 : (BookCrossingUser) model.getUser(user2.getID());
    if (notNullAndEqual(bcUser1.getCity(), bcUser2.getCity())) {
      return 1.0;
    }
    if (notNullAndEqual(bcUser1.getState(), bcUser2.getState())) {
      return 0.5;
    }
    if (notNullAndEqual(bcUser1.getCountry(), bcUser2.getCountry())) {
      return 0.1;
    }
    return 0.0;
  }

  private static boolean notNullAndEqual(Object o1, Object o2) {
    return o1 != null && o1.equals(o2);
  }

  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    // do nothing
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
