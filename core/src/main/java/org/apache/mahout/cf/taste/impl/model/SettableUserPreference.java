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

/**
 * Simple interface that simply exposes a {@link #setUser(User)} method.
 * This helps unify implementations of {@link Preference} which expose this method.
 */
public interface SettableUserPreference extends Preference {

  /**
   * <p>Let this be set by {@link GenericUser} to avoid a circularity problem -- 
   * implementations want a reference to a {@link User} in the constructor, but so does
   * {@link GenericUser}.</p>
   *
   * @param user user whose preference this is
   */
  void setUser(User user);

}