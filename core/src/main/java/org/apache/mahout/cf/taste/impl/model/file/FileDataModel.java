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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>A {@link DataModel} backed by a comma-delimited file. This class assumes that each line of the
 * file contains a user ID, followed by item ID, followed by preferences value, separated by commas.
 * The preference value is assumed to be parseable as a <code>double</code>. The user and item IDs
 * are ready literally as Strings and treated as such in the API. Note that this means that whitespace
 * matters in the data file; they will be treated as part of the ID values.</p>
 *
 * <p>This class is not intended for use with very large amounts of data (over, say, a million rows). For
 * that, a JDBC-backed {@link DataModel} and a database are more appropriate.
 * The file will be periodically reloaded if a change is detected.</p>
 */
public class FileDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(FileDataModel.class);

  private static final Timer timer = new Timer(true);
  private static final long RELOAD_CHECK_INTERVAL_MS = 60L * 1000L;

  private final File dataFile;
  private long lastModified;
  private boolean loaded;
  private DataModel delegate;
  private final ReentrantLock refreshLock;
  private final ReentrantLock reloadLock;

  /**
   * @param dataFile file containing preferences data
   * @throws FileNotFoundException if dataFile does not exist
   */
  public FileDataModel(File dataFile) throws FileNotFoundException {
    if (dataFile == null) {
      throw new IllegalArgumentException("dataFile is null");
    }
    if (!dataFile.exists() || dataFile.isDirectory()) {
      throw new FileNotFoundException(dataFile.toString());
    }

    log.info("Creating FileDataModel for file " + dataFile);

    this.dataFile = dataFile;
    this.lastModified = dataFile.lastModified();
    this.refreshLock = new ReentrantLock();
    this.reloadLock = new ReentrantLock();

    // Schedule next refresh
    timer.schedule(new RefreshTimerTask(), RELOAD_CHECK_INTERVAL_MS, RELOAD_CHECK_INTERVAL_MS);
  }

  protected void reload() {
    try {
      reloadLock.lock();
      Map<String, List<Preference>> data = new FastMap<String, List<Preference>>();

      processFile(data);

      List<User> users = new ArrayList<User>(data.size());
      for (Map.Entry<String, List<Preference>> entries : data.entrySet()) {
        users.add(buildUser(entries.getKey(), entries.getValue()));
      }

      delegate = new GenericDataModel(users);
      loaded = true;

    } finally {
      reloadLock.unlock();
    }
  }

  private void processFile(Map<String, List<Preference>> data) {
    log.info("Reading file info...");
    for (String line : new FileLineIterable(dataFile)) {
      if (line.length() > 0) {
        log.debug("Read line: {}", line);
        processLine(line, data);
      }
    }
  }

  private void processLine(String line, Map<String, List<Preference>> data) {
    int commaOne = line.indexOf((int) ',');
    int commaTwo = line.indexOf((int) ',', commaOne + 1);
    if (commaOne < 0 || commaTwo < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }
    String userID = line.substring(0, commaOne);
    String itemID = line.substring(commaOne + 1, commaTwo);
    double preferenceValue = Double.parseDouble(line.substring(commaTwo + 1));
    List<Preference> prefs = data.get(userID);
    if (prefs == null) {
      prefs = new ArrayList<Preference>();
      data.put(userID, prefs);
    }
    Item item = buildItem(itemID);
      log.debug("Read item '{}' for user ID '{}'", item, userID);
    prefs.add(buildPreference(null, item, preferenceValue));
  }

  private void checkLoaded() throws TasteException {
    if (!loaded) {
      reload();
    }
  }

  public Iterable<? extends User> getUsers() throws TasteException {
    checkLoaded();
    return delegate.getUsers();
  }

  /**
   * @throws NoSuchElementException if there is no such user
   */
  public User getUser(Object id) throws TasteException {
    checkLoaded();
    return delegate.getUser(id);
  }

  public Iterable<? extends Item> getItems() throws TasteException {
    checkLoaded();
    return delegate.getItems();
  }

  public Item getItem(Object id) throws TasteException {
    checkLoaded();
    return delegate.getItem(id);
  }

  public Iterable<? extends Preference> getPreferencesForItem(Object itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItem(itemID);
  }

  public Preference[] getPreferencesForItemAsArray(Object itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItemAsArray(itemID);
  }

  public int getNumItems() throws TasteException {
    checkLoaded();
    return delegate.getNumItems();
  }

  public int getNumUsers() throws TasteException {
    checkLoaded();
    return delegate.getNumUsers();
  }

  public int getNumUsersWithPreferenceFor(Object... itemIDs) throws TasteException {
    checkLoaded();
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  /**
   * @throws UnsupportedOperationException
   */
  public void setPreference(Object userID, Object itemID, double value) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  public void removePreference(Object userID, Object itemID) {
    throw new UnsupportedOperationException();
  }

  public void refresh() {
    if (refreshLock.isLocked()) {
      return;
    }
    try {
      refreshLock.lock();
      reload();
    } finally {
      refreshLock.unlock();
    }

  }

  /**
   * Subclasses may override to return a different {@link User} implementation.
   *
   * @param id user ID
   * @param prefs user preferences
   * @return {@link GenericUser} by default
   */
  protected User buildUser(String id, List<Preference> prefs) {
    return new GenericUser<String>(id, prefs);
  }

  /**
   * Subclasses may override to return a different {@link Item} implementation.
   *
   * @param id item ID
   * @return {@link GenericItem} by default
   */
  protected Item buildItem(String id) {
    return new GenericItem<String>(id);
  }

  /**
   * Subclasses may override to return a different {@link Preference} implementation.
   *
   * @param user {@link User} who expresses the preference
   * @param item preferred {@link Item}
   * @param value preference value
   * @return {@link GenericPreference} by default
   */
  protected Preference buildPreference(User user, Item item, double value) {
    return new GenericPreference(user, item, value);
  }

  @Override
  public String toString() {
    return "FileDataModel[dataFile:" + dataFile + ']';
  }

  private final class RefreshTimerTask extends TimerTask {

    @Override
    public void run() {
      if (loaded) {
        long newModified = dataFile.lastModified();
        if (newModified > lastModified) {
          log.debug("File has changed; reloading...");
          lastModified = newModified;
          reload();
        }
      }
    }
  }

}
