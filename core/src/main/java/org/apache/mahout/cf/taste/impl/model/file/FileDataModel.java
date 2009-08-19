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
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FileLineIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
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
 * <p>Note that it's all-or-nothing -- all of the items in the file must express no preference, or the all must.
 * These cannot be mixed. Put another way there will always be the same number of delimiters on every line of the
 * file!</p>
 *
 * <p>This class is not intended for use with very large amounts of data (over, say, tens of millions of rows). For
 * that, a JDBC-backed {@link DataModel} and a database are more appropriate.</p>
 *
 * <p>It is possible and likely useful to subclass this class and customize its behavior to accommodate
 * application-specific needs and input formats. See {@link #processLine(String, FastByIDMap, char)} and
 * {@link #processLineWithoutID(String, FastByIDMap, char)}
 */
public class FileDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(FileDataModel.class);

  private static final long MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?
  private static final char COMMENT_CHAR = '#';

  private final File dataFile;
  private long lastModified;
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
        delegate = buildModel();
        loaded = true;
      } catch (IOException ioe) {
        log.warn("Exception while reloading", ioe);
      } finally {
        reloadLock.unlock();
      }
    }
  }

  private DataModel buildModel() throws IOException {
    FileLineIterator iterator = new FileLineIterator(dataFile, false);
    String firstLine = iterator.peek();
    while (firstLine.length() == 0 || firstLine.charAt(0) == COMMENT_CHAR) {
      iterator.next();
      firstLine = iterator.peek();
    }
    char delimiter = determineDelimiter(firstLine);
    boolean hasPrefValues = firstLine.indexOf(delimiter, firstLine.indexOf(delimiter) + 1) >= 0;

    if (hasPrefValues) {
      FastByIDMap<Collection<Preference>> data = new FastByIDMap<Collection<Preference>>();
      processFile(iterator, data, delimiter);
      for (File updateFile : findUpdateFiles()) {
        processFile(new FileLineIterator(updateFile, false), data, delimiter);
      }
      return new GenericDataModel(GenericDataModel.toDataMap(data, true));
    } else {
      FastByIDMap<FastIDSet> data = new FastByIDMap<FastIDSet>();
      processFileWithoutID(iterator, data, delimiter);
      for (File updateFile : findUpdateFiles()) {
        processFileWithoutID(new FileLineIterator(updateFile, false), data, delimiter);
      }
      return new GenericBooleanPrefDataModel(data);
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

  private static char determineDelimiter(String line) {
    char delimiter;
    if (line.indexOf(',') >= 0) {
      delimiter = ',';
    } else if (line.indexOf('\t') >= 0) {
      delimiter = '\t';
    } else {
      throw new IllegalArgumentException("Did not find a delimiter in first line");
    }
    int delimiterCount = 0;
    int lastDelimiter = line.indexOf(delimiter);
    int nextDelimiter;
    while ((nextDelimiter = line.indexOf(delimiter, lastDelimiter + 1)) >= 0) {
      delimiterCount++;
      if (delimiterCount == 3) {
        throw new IllegalArgumentException("More than two delimiters per line");
      }
      if (nextDelimiter == lastDelimiter + 1) {
        // empty field
        throw new IllegalArgumentException("Empty field");
      }
      lastDelimiter = nextDelimiter;
    }
    return delimiter;
  }

  protected void processFile(FileLineIterator dataOrUpdateFileIterator,
                             FastByIDMap<Collection<Preference>> data,
                             char delimiter) {
    log.info("Reading file info...");
    AtomicInteger count = new AtomicInteger();
    while (dataOrUpdateFileIterator.hasNext()) {
      String line = dataOrUpdateFileIterator.next();
      if (line.length() > 0) {
        processLine(line, data, delimiter);
        int currentCount = count.incrementAndGet();
        if (currentCount % 100000 == 0) {
          log.info("Processed {} lines", currentCount);
        }
      }
    }
    log.info("Read lines: {}", count.get());
  }

  /**
   * <p>Reads one line from the input file and adds the data to a {@link Map} data structure which maps user IDs to
   * preferences. This assumes that each line of the input file corresponds to one preference. After reading a line and
   * determining which user and item the preference pertains to, the method should look to see if the data contains a
   * mapping for the user ID already, and if not, add an empty {@link List} of {@link Preference}s to the data.</p>
   *
   * <p>Note that if the line is empty or begins with '#' it will be ignored as a comment.</p>
   *
   * @param line      line from input data file
   * @param data      all data read so far, as a mapping from user IDs to preferences
   */
  protected void processLine(String line, FastByIDMap<Collection<Preference>> data, char delimiter) {

    if (line.length() == 0 || line.charAt(0) == COMMENT_CHAR) {
      return;
    }

    int delimiterOne = line.indexOf((int) delimiter);
    int delimiterTwo = line.indexOf((int) delimiter, delimiterOne + 1);
    if (delimiterOne < 0 || delimiterTwo < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }

    long userID = Long.parseLong(line.substring(0, delimiterOne));
    long itemID = Long.parseLong(line.substring(delimiterOne + 1, delimiterTwo));
    String preferenceValueString = line.substring(delimiterTwo + 1);

    if (transpose) {
      long tmp = userID;
      userID = itemID;
      itemID = tmp;
    }
    Collection<Preference> prefs = data.get(userID);
    if (prefs == null) {
      prefs = new ArrayList<Preference>(2);
      data.put(userID, prefs);
    }

    if (preferenceValueString.length() == 0) {
      // remove pref
      Iterator<Preference> prefsIterator = prefs.iterator();
      while (prefsIterator.hasNext()) {
        Preference pref = prefsIterator.next();
        if (pref.getItemID() == itemID) {
          prefsIterator.remove();
          break;
        }
      }
    } else {
      float preferenceValue = Float.parseFloat(preferenceValueString);
      prefs.add(new GenericPreference(userID, itemID, preferenceValue));
    }
  }

  protected void processFileWithoutID(FileLineIterator dataOrUpdateFileIterator,
                                      FastByIDMap<FastIDSet> data,
                                      char delimiter) {
    log.info("Reading file info...");
    AtomicInteger count = new AtomicInteger();
    while (dataOrUpdateFileIterator.hasNext()) {
      String line = dataOrUpdateFileIterator.next();
      if (line.length() > 0) {
        processLineWithoutID(line, data, delimiter);
        int currentCount = count.incrementAndGet();
        if (currentCount % 100000 == 0) {
          log.info("Processed {} lines", currentCount);
        }
      }
    }
    log.info("Read lines: {}", count.get());
  }

  protected void processLineWithoutID(String line, FastByIDMap<FastIDSet> data, char delimiter) {

    if (line.length() == 0 || line.charAt(0) == COMMENT_CHAR) {
      return;
    }

    int delimiterOne = line.indexOf((int) delimiter);
    if (delimiterOne < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }

    long userID = Long.parseLong(line.substring(0, delimiterOne));
    long itemID = Long.parseLong(line.substring(delimiterOne + 1));

    if (transpose) {
      long tmp = userID;
      userID = itemID;
      itemID = tmp;
    }
    FastIDSet itemIDs = data.get(userID);
    if (itemIDs == null) {
      itemIDs = new FastIDSet(2);
      data.put(userID, itemIDs);
    }
    itemIDs.add(itemID);
  }

  private void checkLoaded() {
    if (!loaded) {
      reload();
    }
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    checkLoaded();
    return delegate.getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesFromUser(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return delegate.getItemIDsFromUser(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    checkLoaded();
    return delegate.getItemIDs();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    checkLoaded();
    return delegate.getPreferencesForItem(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceValue(userID, itemID);
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
  public int getNumUsersWithPreferenceFor(long... itemIDs) throws TasteException {
    checkLoaded();
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  /**
   * Note that this method only updates the in-memory preference data that this {@link FileDataModel} maintains; it does
   * not modify any data on disk. Therefore any updates from this method are only temporary, and lost when data is
   * reloaded from a file. This method should also be considered relatively slow.
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    checkLoaded();
    delegate.setPreference(userID, itemID, value);
  }

  /** See the warning at {@link #setPreference(long, long, float)}. */
  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
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

  @Override
  public String toString() {
    return "FileDataModel[dataFile:" + dataFile + ']';
  }



}
