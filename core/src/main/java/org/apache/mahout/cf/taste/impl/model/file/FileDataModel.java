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
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.model.BooleanPrefUser;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>A {@link DataModel} backed by a comma-delimited file. This class typically expects a file where each line contains
 * a user ID, followed by item ID, followed by preferences value, separated by commas. You may also use tabs.</p>
 *
 * <p>The preference value is assumed to be parseable as a <code>double</code>. The user and item IDs are ready
 * literally as Strings and treated as such in the API. Note that this means that whitespace matters in the data file;
 * they will be treated as part of the ID values.</p>
 *
 * <p>This class will reload data from the data file when {@link #refresh(Collection)} is called, unless the file has
 * been reloaded very recently already.</p>
 *
 * <p>This class will also look for update "delta" files in the same directory, with file names that start the same way
 * (up to the first period). These files should have the same format, and provide updated data that supersedes what is
 * in the main data file. This is a mechanism that allows an application to push updates to {@link FileDataModel}
 * without re-copying the entire data file.</p>
 *
 * <p>The line may contain a blank preference value (e.g. "123,ABC,"). This is interpreted to mean "delete preference",
 * and is only useful in the context of an update delta file (see above). Note that if the line is empty or begins with
 * '#' it will be ignored as a comment.</p>
 *
 * <p>Finally, for application that have no notion of a preference value (that is, the user simply expresses a
 * preference for an item, but no degree of preference), the caller can simply omit the third token in each line
 * altogether -- for example, "123,ABC".</p>
 *
 * <p>This class is not intended for use with very large amounts of data (over, say, tens of millions of rows). For
 * that, a JDBC-backed {@link DataModel} and a database are more appropriate.</p>
 *
 * <p>It is possible and likely useful to subclass this class and customize its behavior to accommodate
 * application-specific needs and input formats. See {@link #processLine(String, Map)},
 * {@link #buildUser(String, List)} and {@link #buildPreference(User, Comparable, double)}.</p>
 */
public class FileDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(FileDataModel.class);

  private static final long MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?
  private static final char UNKNOWN_DELIMITER = '\0';

  private final File dataFile;
  private long lastModified;
  private char delimiter;
  private boolean loaded;
  private DataModel delegate;
  private final ReentrantLock reloadLock;
  private final boolean transpose;

  /**
   * @param dataFile file containing preferences data. If file is compressed (and name ends in .gz or .zip accordingly)
   *                 it will be decompressed as it is read)
   * @throws FileNotFoundException if dataFile does not exist
   */
  public FileDataModel(File dataFile) throws FileNotFoundException {
    this(dataFile, false);
  }

  public FileDataModel(File dataFile, boolean transpose) throws FileNotFoundException {
    if (dataFile == null) {
      throw new IllegalArgumentException("dataFile is null");
    }
    if (!dataFile.exists() || dataFile.isDirectory()) {
      throw new FileNotFoundException(dataFile.toString());
    }

    this.delimiter = UNKNOWN_DELIMITER;

    log.info("Creating FileDataModel for file " + dataFile);

    this.dataFile = dataFile.getAbsoluteFile();
    this.lastModified = dataFile.lastModified();
    this.reloadLock = new ReentrantLock();
    this.transpose = transpose;
  }

  public File getDataFile() {
    return dataFile;
  }

  protected void reload() {
    if (!reloadLock.isLocked()) {
      reloadLock.lock();
      try {
        Map<String, List<Preference>> data = new FastMap<String, List<Preference>>();

        processFile(dataFile, data);
        for (File updateFile : findUpdateFiles()) {
          processFile(updateFile, data);
        }

        delegate = new GenericDataModel(new UserIterableOverData(data));
        loaded = true;

      } finally {
        reloadLock.unlock();
      }
    }
  }

  /**
   * Finds update delta files in the same directory as the data file. This finds any file whose name starts the same way
   * as the data file (up to first period) but isn't the data file itself. For example, if the data file is
   * /foo/data.txt.gz, you might place update files at /foo/data.1.txt.gz, /foo/data.2.txt.gz, etc.
   */
  private Iterable<File> findUpdateFiles() {
    String dataFileName = dataFile.getName();
    int period = dataFileName.indexOf('.');
    String startName = period < 0 ? dataFileName : dataFileName.substring(0, period);
    File parentDir = dataFile.getParentFile();
    List<File> updateFiles = new ArrayList<File>();
    for (File updateFile : parentDir.listFiles()) {
      String updateFileName = updateFile.getName();
      if (updateFileName.startsWith(startName) && !updateFileName.equals(dataFileName)) {
        updateFiles.add(updateFile);
      }
    }
    Collections.sort(updateFiles);
    return updateFiles;
  }

  protected void processFile(File dataOrUpdateFile, Map<String, List<Preference>> data) {
    log.info("Reading file info...");
    AtomicInteger count = new AtomicInteger();
    for (String line : new FileLineIterable(dataOrUpdateFile, false)) {
      if (line.length() > 0) {
        if (log.isDebugEnabled()) {
          log.debug("Read line: {}", line);
        }
        if (delimiter == UNKNOWN_DELIMITER) {
          delimiter = determineDelimiter(line);
        }
        processLine(line, data);
        int currentCount = count.incrementAndGet();
        if (currentCount % 100000 == 0) {
          log.info("Processed {} lines", currentCount);
        }
      }
    }
    log.info("Read lines: " + count.get());
  }

  private static char determineDelimiter(String line) {
    if (line.indexOf(',') >= 0) {
      return ',';
    }
    if (line.indexOf('\t') >= 0) {
      return '\t';
    }
    throw new IllegalArgumentException("Did not find a delimiter in first line");
  }

  /**
   * <p>Reads one line from the input file and adds the data to a {@link Map} data structure which maps user IDs to
   * preferences. This assumes that each line of the input file corresponds to one preference. After reading a line and
   * determining which user and item the preference pertains to, the method should look to see if the data contains a
   * mapping for the user ID already, and if not, add an empty {@link List} of {@link Preference}s to the data.</p>
   *
   * <p>The method should use {@link #buildPreference(User, Comparable, double)} to
   * build {@link Preference} objects as needed.</p>
   *
   * <p>Note that if the line is empty or begins with '#' it will be ignored as a comment.</p>
   *
   * @param line      line from input data file
   * @param data      all data read so far, as a mapping from user IDs to preferences
   * @see #buildPreference(User, Comparable, double)
   */
  protected void processLine(String line, Map<String, List<Preference>> data) {

    if (line.length() == 0 || line.charAt(0) == '#') {
      return;
    }

    int delimiterOne = line.indexOf((int) delimiter);
    if (delimiterOne < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }
    int delimiterTwo = line.indexOf((int) delimiter, delimiterOne + 1);

    String userID = line.substring(0, delimiterOne);
    String itemID;
    String preferenceValueString;
    if (delimiterTwo >= 0) {
      itemID = line.substring(delimiterOne + 1, delimiterTwo);
      preferenceValueString = line.substring(delimiterTwo + 1);
    } else {
      itemID = line.substring(delimiterOne + 1);
      preferenceValueString = null;
    }
    if (transpose) {
      String tmp = userID;
      userID = itemID;
      itemID = tmp;
    }
    List<Preference> prefs = data.get(userID);
    if (prefs == null) {
      prefs = new ArrayList<Preference>(2);
      data.put(userID, prefs);
    }

    if (preferenceValueString != null && preferenceValueString.length() == 0) {
      // remove pref
      Iterator<Preference> prefsIterator = prefs.iterator();
      while (prefsIterator.hasNext()) {
        Preference pref = prefsIterator.next();
        if (pref.getItemID().equals(itemID)) {
          prefsIterator.remove();
          break;
        }
      }
    } else {
      // add pref -- assume it does not already exist
      if (preferenceValueString == null) {
        prefs.add(new BooleanPreference(null, itemID));
      } else {
        double preferenceValue = Double.parseDouble(preferenceValueString);
        prefs.add(buildPreference(null, itemID, preferenceValue));
      }
    }
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
  public User getUser(Comparable<?> id) throws TasteException {
    checkLoaded();
    return delegate.getUser(id);
  }

  @Override
  public Iterable<Comparable<?>> getItemIDs() throws TasteException {
    checkLoaded();
    return delegate.getItemIDs();
  }

  @Override
  public Iterable<? extends Preference> getPreferencesForItem(Comparable<?> itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItem(itemID);
  }

  @Override
  public Preference[] getPreferencesForItemAsArray(Comparable<?> itemID) throws TasteException {
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
  public int getNumUsersWithPreferenceFor(Comparable<?>... itemIDs) throws TasteException {
    checkLoaded();
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  /**
   * Note that this method only updates the in-memory preference data that this {@link FileDataModel} maintains; it does
   * not modify any data on disk. Therefore any updates from this method are only temporary, and lost when data is
   * reloaded from a file. This method should also be considered relatively slow.
   */
  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, double value) throws TasteException {
    checkLoaded();
    delegate.setPreference(userID, itemID, value);
  }

  /** See the warning at {@link #setPreference(Comparable, Comparable, double)}. */
  @Override
  public void removePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
    checkLoaded();
    delegate.removePreference(userID, itemID);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    long mostRecentModification = dataFile.lastModified();
    for (File updateFile : findUpdateFiles()) {
      mostRecentModification = Math.max(mostRecentModification, updateFile.lastModified());
    }
    if (mostRecentModification > lastModified + MIN_RELOAD_INTERVAL_MS) {
      log.debug("File has changed; reloading...");
      lastModified = mostRecentModification;
      reload();
    }
  }

  /**
   * Subclasses may override to return a different {@link User} implementation. The default implemenation always builds
   * a new {@link GenericUser}. This may not be desirable; it may be better to return an existing {@link User} object in
   * some applications rather than create a new object.
   *
   * @param id    user ID
   * @param prefs user preferences
   * @return {@link GenericUser} by default, or, a {@link BooleanPrefUser} if the prefs supplied are in fact {@link
   *         BooleanPreference}s
   */
  protected User buildUser(String id, List<Preference> prefs) {
    if (!prefs.isEmpty() && prefs.get(0) instanceof BooleanPreference) {
      // If first is a BooleanPreference, assuming all are, so, want to use BooleanPrefUser
      FastSet<Comparable<?>> itemIDs = new FastSet<Comparable<?>>(prefs.size());
      for (Preference pref : prefs) {
        itemIDs.add(pref.getItemID());
      }
      itemIDs.rehash();
      return new BooleanPrefUser(id, itemIDs);
    }
    return new GenericUser(id, prefs);
  }

  /**
   * Subclasses may override to return a different {@link Preference} implementation. The default implementation builds
   * a new {@link GenericPreference}.
   *
   * @param user  {@link User} who expresses the preference
   * @param itemID  preferred item
   * @param value preference value
   * @return {@link GenericPreference} by default
   */
  protected Preference buildPreference(User user, Comparable<?> itemID, double value) {
    return new GenericPreference(user, itemID, value);
  }

  @Override
  public String toString() {
    return "FileDataModel[dataFile:" + dataFile + ']';
  }


  private final class UserIterableOverData implements Iterable<User> {
    private final Map<String, List<Preference>> data;

    private UserIterableOverData(Map<String, List<Preference>> data) {
      this.data = data;
    }

    @Override
    public Iterator<User> iterator() {
      return new UserIteratorOverData(data.entrySet().iterator());
    }
  }

  private final class UserIteratorOverData implements Iterator<User> {
    private final Iterator<Map.Entry<String, List<Preference>>> dataIterator;

    private UserIteratorOverData(Iterator<Map.Entry<String, List<Preference>>> dataIterator) {
      this.dataIterator = dataIterator;
    }

    @Override
    public boolean hasNext() {
      return dataIterator.hasNext();
    }

    @Override
    public User next() {
      Map.Entry<String, List<Preference>> datum = dataIterator.next();
      String key = datum.getKey();
      List<Preference> value = datum.getValue();
      dataIterator.remove();
      return buildUser(key, value);
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

}
