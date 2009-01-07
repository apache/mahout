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
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.Item;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public final class JesterDataModel extends FileDataModel {

  private int userBeingRead;

  public JesterDataModel() throws IOException {
    this(GroupLensDataModel.readResourceToTempFile("/org/apache/mahout/cf/taste/example/jester/jester-data-1.csv"));
  }

  /**
   * @param ratingsFile Jester ratings file in CSV format
   * @throws java.io.IOException if an error occurs while reading or writing files
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
  protected void processLine(String line, Map<String, List<Preference>> data, Map<String, Item> itemCache) {
    String userID = String.valueOf(userBeingRead);
    String[] jokePrefs = line.split(",");
    List<Preference> prefs = new ArrayList<Preference>(101);
    for (int itemIDNum = 1; itemIDNum < jokePrefs.length; itemIDNum++) { // yes skip first one, just a count
      String jokePref = jokePrefs[itemIDNum];
      if (!"99".equals(jokePref)) {
        double jokePrefValue = Double.parseDouble(jokePref);        
        String itemID = String.valueOf(itemIDNum);
        Item item = itemCache.get(itemID);
        if (item == null) {
          item = buildItem(itemID);
          itemCache.put(itemID, item);
        }
        prefs.add(new GenericPreference(null, item, jokePrefValue));
      }
    }
    data.put(userID, prefs);
    userBeingRead++;
  }

  @Override
  public String toString() {
    return "JesterDataModel";
  }

}