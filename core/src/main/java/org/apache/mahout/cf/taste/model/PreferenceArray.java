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

package org.apache.mahout.cf.taste.model;

/**
 * An alternate representation of an array of {@link Preference}. Implementations, in theory, can produce a more
 * memory-efficient representation. This is not used yet.
 */
public interface PreferenceArray {

  Preference get(int i);

  void set(int i, Preference pref);

  User getUser(int i);

  void setUser(int i, User user);

  Comparable<?> getItemID(int i);

  void setItemID(int i, Comparable<?> itemID);

  double getValue(int i);

  void setValue(int i, double value);

}