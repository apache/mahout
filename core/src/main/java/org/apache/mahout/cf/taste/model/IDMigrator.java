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

import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>Mahout 0.2 changed the framework to operate only in terms of numeric (long) ID values
 * for users and items. This is, obviously, not compatible with applications that used other
 * key types -- most commonly {@link String}. Implementation of this class provide support for
 * mapping String to longs and vice versa in order to provide a smoother migration path to
 * applications that must still use strings as IDs.</p>
 *
 * <p>The mapping from strings to 64-bit numeric values is fixed here, to provide a standard
 * implementation that is 'portable' or reproducible outside the framework easily. See
 * {@link #toLongID(String)}.</p>
 *
 * <p>Because this mapping is deterministically computable, it does not need to be stored. Indeed,
 * subclasses' job is to store the reverse mapping. There are an infinite number of strings but only
 * a fixed number of longs, so, it is possible for two strings to map to the same value. Subclasses
 * do not treat this as an error but rather retain only the most recent mapping, overwriting a previous
 * mapping. The probability of collision in a 64-bit space is quite small, but not zero. However,
 * in the context of a collaborative filtering problem, the consequence of a collision is small, at worst
 * -- perhaps one user receives another recommendations.</p>
 *
 * @since 0.2
 */
public interface IDMigrator {

  /**
   * @return the top 8 bytes of the MD5 hash of the bytes of the given {@link String}'s UTF-8 encoding as a long.
   *  The reverse mapping is also stored.
   * @throws TasteException if an error occurs while storing the mapping
   */
  long toLongID(String stringID) throws TasteException;

  /**
   * @return the string ID most recently associated with the given long ID, or null if doesn't exist
   * @throws TasteException if an error occurs while retrieving the mapping
   */
  String toStringID(long longID) throws TasteException;

  /**
   * Make the mapping aware of the given string IDs.
   *
   * @throws TasteException if an error occurs while storing the mappings
   */
  void initialize(Iterable<String> stringIDs) throws TasteException;

}
