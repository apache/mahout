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

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.model.UpdatableIDMigrator;

/**
 * Implementation which stores the reverse long-to-String mapping in memory.
 */
public final class MemoryIDMigrator extends AbstractIDMigrator implements UpdatableIDMigrator {
  
  private final FastByIDMap<String> longToString;
  
  public MemoryIDMigrator() {
    this.longToString = new FastByIDMap<String>(100);
  }
  
  @Override
  public void storeMapping(long longID, String stringID) {
    synchronized (longToString) {
      longToString.put(longID, stringID);
    }
  }
  
  @Override
  public String toStringID(long longID) {
    synchronized (longToString) {
      return longToString.get(longID);
    }
  }

  @Override
  public void initialize(Iterable<String> stringIDs) {
    for (String stringID : stringIDs) {
      storeMapping(toLongID(stringID), stringID);
    }
  }
  
}
