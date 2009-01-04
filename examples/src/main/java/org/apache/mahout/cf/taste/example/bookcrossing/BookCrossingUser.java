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

import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.Preference;

import java.util.List;

final class BookCrossingUser extends GenericUser<String> {

  private final String city;
  private final String state;
  private final String country;
  private final Integer age;

  BookCrossingUser(String id, List<Preference> prefs, String city, String state, String country, Integer age) {
    super(id, prefs);
    this.city = city;
    this.state = state;
    this.country = country;
    this.age = age;
  }

  String getCity() {
    return city;
  }

  String getState() {
    return state;
  }

  String getCountry() {
    return country;
  }

  Integer getAge() {
    return age;
  }

}