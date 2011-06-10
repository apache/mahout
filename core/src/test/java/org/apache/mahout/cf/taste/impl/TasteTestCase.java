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

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;

public abstract class TasteTestCase extends MahoutTestCase {

  public static DataModel getDataModel(long[] userIDs, Double[][] prefValues) {
    FastByIDMap<PreferenceArray> result = new FastByIDMap<PreferenceArray>();
    for (int i = 0; i < userIDs.length; i++) {
      List<Preference> prefsList = Lists.newArrayList();
      for (int j = 0; j < prefValues[i].length; j++) {
        if (prefValues[i][j] != null) {
          prefsList.add(new GenericPreference(userIDs[i], j, prefValues[i][j].floatValue()));
        }
      }
      if (!prefsList.isEmpty()) {
        result.put(userIDs[i], new GenericUserPreferenceArray(prefsList));
      }
    }
    return new GenericDataModel(result);
  }

  protected static DataModel getDataModel() {
    return getDataModel(
            new long[] {1, 2, 3, 4},
            new Double[][] {
                    {0.1, 0.3},
                    {0.2, 0.3, 0.3},
                    {0.4, 0.3, 0.5},
                    {0.7, 0.3, 0.8},
            });
  }

  protected static boolean arrayContains(long[] array, long value) {
    for (long l : array) {
      if (l == value) {
        return true;
      }
    }
    return false;
  }

  protected static void writeLines(File file, String... lines) throws FileNotFoundException {
    PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file), Charsets.UTF_8));
    try {
      for (String line : lines) {
        writer.println(line);
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
}
