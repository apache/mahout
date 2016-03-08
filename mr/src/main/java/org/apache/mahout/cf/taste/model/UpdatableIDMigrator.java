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

public interface UpdatableIDMigrator extends IDMigrator {
  
  /**
   * Stores the reverse long-to-String mapping in some kind of backing store. Note that this must be called
   * directly (or indirectly through {@link #initialize(Iterable)}) for every String that might be encountered
   * in the application, or else the mapping will not be known.
   *
   * @param longID
   *          long ID
   * @param stringID
   *          string ID that maps to/from that long ID
   * @throws TasteException
   *           if an error occurs while saving the mapping
   */
  void storeMapping(long longID, String stringID) throws TasteException;

  /**
   * Make the mapping aware of the given string IDs. This must be called initially before the implementation
   * is used, or else it will not be aware of reverse long-to-String mappings.
   *
   * @throws TasteException
   *           if an error occurs while storing the mappings
   */
  void initialize(Iterable<String> stringIDs) throws TasteException;
  
}
