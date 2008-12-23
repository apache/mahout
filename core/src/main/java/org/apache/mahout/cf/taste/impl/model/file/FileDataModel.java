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
import java.util.Collection;
import java.util.List;
import java.util.Map;
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
 *
 * <p>It is possible and likely useful to subclass this class and customize its behavior to accommodate
 * application-specific needs and input formats. See {@link #processLine(String, Map, Map)},
 * {@link #buildItem(String)}, {@link #buildUser(String, List)}
 * and {@link #buildPreference(User, Item, double)}.</p>
 */
public class FileDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(FileDataModel.class);

  private static final Timer timer = new Timer(true);
  private static final long RELOAD_CHECK_INTERVAL_MS = 60L * 1000L;

  private final File dataFile;
  private long lastModified;
  private boolean loaded;
  private DataModel delegate;
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
    this.reloadLock = new ReentrantLock();

    // Schedule next refresh
    timer.schedule(new RefreshTimerTask(), RELOAD_CHECK_INTERVAL_MS, RELOAD_CHECK_INTERVAL_MS);
  }

  protected void reload() {
    reloadLock.lock();    
    try {
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
    Map<String, Item> itemCache = new FastMap<String, Item>(1001);
    for (String line : new FileLineIterable(dataFile)) {
      if (line.length() > 0) {
        log.debug("Read line: {}", line);
        processLine(line, data, itemCache);
      }
    }
  }

  /**
   * <p>Reads one line from the input file and adds the data to a {@link Map} data structure
   * which maps user IDs to preferences. This assumes that each line of the input file
   * corresponds to one preference. After reading a line and determining which user and item
   * the preference pertains to, the method should look to see if the data contains a mapping
   * for the user ID already, and if not, add an empty {@link List} of {@link Preference}s to
   * the data.</p>
   *
   * <p>The method should use {@link #buildItem(String)} to create an {@link Item} representing
   * the item in question if needed, and use {@link #buildPreference(User, Item, double)} to
   * build {@link Preference} objects as needed.</p>
   *
   * @param line line from input data file
   * @param data all data read so far, as a mapping from user IDs to preferences
   * @see #buildPreference(User, Item, double)
   * @see #buildItem(String)
   */
  protected void processLine(String line, Map<String, List<Preference>> data, Map<String, Item> itemCache) {
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
    Item item = itemCache.get(itemID);
    if (item == null) {
      item = buildItem(itemID);
      itemCache.put(itemID, item);
    }
    log.debug("Read item '{}' for user ID '{}'", item, userID);
    prefs.add(buildPreference(null, item, preferenceValue));
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

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void setPreference(Object userID, Object itemID, double value) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void removePreference(Object userID, Object itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    reload();
  }

  /**
   * Subclasses may override to return a different {@link User} implementation.
   * The default implemenation always builds a new {@link GenericUser}. This may not
   * be desirable; it may be better to return an existing {@link User} object
   * in some applications rather than create a new object.
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
   * The default implementation always builds a new {@link GenericItem}; likewise
   * it may be better here to return an existing object encapsulating the item
   * instead.
   *
   * @param id item ID
   * @return {@link GenericItem} by default
   */
  protected Item buildItem(String id) {
    return new GenericItem<String>(id);
  }

  /**
   * Subclasses may override to return a different {@link Preference} implementation.
   * The default implementation builds a new {@link GenericPreference}.
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
