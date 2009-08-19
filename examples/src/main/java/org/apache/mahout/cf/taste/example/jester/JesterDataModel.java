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

package org.apache.mahout.cf.taste.example.jester;

import org.apache.mahout.cf.taste.example.grouplens.GroupLensDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.model.Preference;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

public final class JesterDataModel extends FileDataModel {

  private long userBeingRead;

  public JesterDataModel() throws IOException {
    this(GroupLensDataModel.readResourceToTempFile("/org/apache/mahout/cf/taste/example/jester/jester-data-1.csv"));
  }

  /**
   * @param ratingsFile Jester ratings file in CSV format
   * @throws IOException if an error occurs while reading or writing files
   */
  public JesterDataModel(File ratingsFile) throws IOException {
    super(ratingsFile);
  }

  @Override
  public void reload() {
    userBeingRead = 0;
    super.reload();
  }

  @Override
  protected void processLine(String line, FastByIDMap<Collection<Preference>> data, char delimiter) {
    String[] jokePrefs = line.split(",");
    int count = Integer.parseInt(jokePrefs[0]);
    Collection<Preference> prefs = new ArrayList<Preference>(count);
    for (int itemID = 1; itemID < jokePrefs.length; itemID++) { // yes skip first one, just a count
      String jokePref = jokePrefs[itemID];
      if (!"99".equals(jokePref)) {
        float jokePrefValue = Float.parseFloat(jokePref);
        prefs.add(new GenericPreference(userBeingRead, itemID, jokePrefValue));
      }
    }
    data.put(userBeingRead, prefs);
    userBeingRead++;
  }

}