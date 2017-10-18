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

package org.apache.mahout.cf.taste.example.kddcup;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;

public final class DataFileIterable implements Iterable<Pair<PreferenceArray,long[]>> {

  private final File dataFile;

  public DataFileIterable(File dataFile) {
    this.dataFile = dataFile;
  }

  @Override
  public Iterator<Pair<PreferenceArray, long[]>> iterator() {
    try {
      return new DataFileIterator(dataFile);
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }
 
}
