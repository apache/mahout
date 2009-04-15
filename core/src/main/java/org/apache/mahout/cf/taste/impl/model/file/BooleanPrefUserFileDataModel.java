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

package org.apache.mahout.cf.taste.impl.model.file;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.BooleanPrefUser;
import org.apache.mahout.cf.taste.impl.model.BooleanUserGenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A variant on {@link FileDataModel} which uses the "boolean" classes like {@link BooleanPrefUser}.
 */
public class BooleanPrefUserFileDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(BooleanPrefUserFileDataModel.class);

  private final File dataFile;
  private boolean loaded;
  private DataModel delegate;
  private final ReentrantLock reloadLock;

  /**
   * @param dataFile file containing preferences data
   * @throws java.io.FileNotFoundException if dataFile does not exist
   */
  public BooleanPrefUserFileDataModel(File dataFile) throws FileNotFoundException {
    if (dataFile == null) {
      throw new IllegalArgumentException("dataFile is null");
    }
    if (!dataFile.exists() || dataFile.isDirectory()) {
      throw new FileNotFoundException(dataFile.toString());
    }

    log.info("Creating FileDataModel for file " + dataFile);

    this.dataFile = dataFile;
    this.reloadLock = new ReentrantLock();
  }

  protected void reload() {
    reloadLock.lock();
    try {
      Map<String, FastSet<Object>> data = new FastMap<String, FastSet<Object>>();

      processFile(data);

      List<User> users = new ArrayList<User>(data.size());
      for (Map.Entry<String, FastSet<Object>> entries : data.entrySet()) {
        users.add(buildUser(entries.getKey(), entries.getValue()));
      }

      delegate = new BooleanUserGenericDataModel(users);
      loaded = true;

    } finally {
      reloadLock.unlock();
    }
  }

  private void processFile(Map<String, FastSet<Object>> data) {
    log.info("Reading file info...");
    for (String line : new FileLineIterable(dataFile, false)) {
      if (line.length() > 0) {
        log.debug("Read line: {}", line);
        processLine(line, data);
      }
    }
    log.info("Done reading file: " + data.size());
  }

  /**
   * <p>Reads one line from the input file and adds the data to a {@link java.util.Map} data structure
   * which maps user IDs to preferences. This assumes that each line of the input file
   * corresponds to one preference. After reading a line and determining which user and item
   * the preference pertains to, the method should look to see if the data contains a mapping
   * for the user ID already, and if not, add an empty {@link java.util.List} of {@link org.apache.mahout.cf.taste.model.Preference}s to
   * the data.</p>
   *
   * <p>The method should use {@link #buildItem(String)} to create an {@link org.apache.mahout.cf.taste.model.Item} representing
   * the item in question if needed, and use {@link #buildPreference(org.apache.mahout.cf.taste.model.User, org.apache.mahout.cf.taste.model.Item, double)} to
   * build {@link org.apache.mahout.cf.taste.model.Preference} objects as needed.</p>
   *
   * @param line line from input data file
   * @param data all data read so far, as a mapping from user IDs to preferences
   * @see #buildPreference(org.apache.mahout.cf.taste.model.User, org.apache.mahout.cf.taste.model.Item, double)
   * @see #buildItem(String)
   */
  protected void processLine(String line, Map<String, FastSet<Object>> data) {
    int commaOne = line.indexOf((int) ',');
    int commaTwo = line.indexOf((int) ',', commaOne + 1);
    if (commaOne < 0 || commaTwo < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }
    String userID = line.substring(0, commaOne);
    String itemID = line.substring(commaOne + 1, commaTwo);
    FastSet<Object> prefs = data.get(userID);
    if (prefs == null) {
      prefs = new FastSet<Object>();
      data.put(userID, prefs);
    }
    prefs.add(itemID);
    log.debug("Read item '{}' for user ID '{}'", itemID, userID);
  }

  private void checkLoaded() {
    if (!loaded) {
      reload();
    }
  }

  @Override
  public Iterable<? extends User> getUsers() throws TasteException {
    checkLoaded();
    return delegate.getUsers();
  }

  @Override
  public User getUser(Object id) throws TasteException {
    checkLoaded();
    return delegate.getUser(id);
  }

  @Override
  public Iterable<? extends Item> getItems() throws TasteException {
    checkLoaded();
    return delegate.getItems();
  }

  @Override
  public Item getItem(Object id) throws TasteException {
    checkLoaded();
    return delegate.getItem(id);
  }

  @Override
  public Iterable<? extends Preference> getPreferencesForItem(Object itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItem(itemID);
  }

  @Override
  public Preference[] getPreferencesForItemAsArray(Object itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItemAsArray(itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
    checkLoaded();
    return delegate.getNumItems();
  }

  @Override
  public int getNumUsers() throws TasteException {
    checkLoaded();
    return delegate.getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(Object... itemIDs) throws TasteException {
    checkLoaded();
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  @Override
  public void setPreference(Object userID, Object itemID, double value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void removePreference(Object userID, Object itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    reload();
  }

  protected User buildUser(String id, FastSet<Object> prefs) {
    return new BooleanPrefUser<String>(id, prefs);
  }

  protected Item buildItem(String id) {
    return new GenericItem<String>(id);
  }

  protected Preference buildPreference(User user, Item item, double value) {
    return new GenericPreference(user, item, value);
  }

  @Override
  public String toString() {
    return "BooleanPrefUserFileDataModel[dataFile:" + dataFile + ']';
  }
}
