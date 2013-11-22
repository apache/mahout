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
import java.util.Collection;
import java.util.Iterator;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.SamplingIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>An {@link DataModel} which reads into memory any of the KDD Cup's rating files; it is really
 * meant for use with training data in the files trainIdx{1,2}}.txt.
 * See http://kddcup.yahoo.com/.</p>
 *
 * <p>Timestamps in the data set are relative to some unknown point in time, for anonymity. They are assumed
 * to be relative to the epoch, time 0, or January 1 1970, for purposes here.</p>
 */
public final class KDDCupDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(KDDCupDataModel.class);

  private final File dataFileDirectory;
  private final DataModel delegate;

  /**
   * @param dataFile training rating file
   */
  public KDDCupDataModel(File dataFile) throws IOException {
    this(dataFile, false, 1.0);
  }

  /**
   * @param dataFile training rating file
   * @param storeDates if true, dates are parsed and stored, otherwise not
   * @param samplingRate percentage of users to keep; can be used to reduce memory requirements
   */
  public KDDCupDataModel(File dataFile, boolean storeDates, double samplingRate) throws IOException {

    Preconditions.checkArgument(!Double.isNaN(samplingRate) && samplingRate > 0.0 && samplingRate <= 1.0,
        "Must be: 0.0 < samplingRate <= 1.0");

    dataFileDirectory = dataFile.getParentFile();

    Iterator<Pair<PreferenceArray,long[]>> dataIterator = new DataFileIterator(dataFile);
    if (samplingRate < 1.0) {
      dataIterator = new SamplingIterator<Pair<PreferenceArray,long[]>>(dataIterator, samplingRate);
    }

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();
    FastByIDMap<FastByIDMap<Long>> timestamps = new FastByIDMap<FastByIDMap<Long>>();

    while (dataIterator.hasNext()) {

      Pair<PreferenceArray,long[]> pair = dataIterator.next();
      PreferenceArray userPrefs = pair.getFirst();
      long[] timestampsForPrefs = pair.getSecond();

      userData.put(userPrefs.getUserID(0), userPrefs);
      if (storeDates) {
        FastByIDMap<Long> itemTimestamps = new FastByIDMap<Long>();
        for (int i = 0; i < timestampsForPrefs.length; i++) {
          long timestamp = timestampsForPrefs[i];
          if (timestamp > 0L) {
            itemTimestamps.put(userPrefs.getItemID(i), timestamp);
          }
        }
      }

    }

    if (storeDates) {
      delegate = new GenericDataModel(userData, timestamps);
    } else {
      delegate = new GenericDataModel(userData);
    }

    Runtime runtime = Runtime.getRuntime();
    log.info("Loaded data model in about {}MB heap", (runtime.totalMemory() - runtime.freeMemory()) / 1000000);
  }

  public File getDataFileDirectory() {
    return dataFileDirectory;
  }

  public static File getTrainingFile(File dataFileDirectory) {
    return getFile(dataFileDirectory, "trainIdx");
  }

  public static File getValidationFile(File dataFileDirectory) {
    return getFile(dataFileDirectory, "validationIdx");
  }

  public static File getTestFile(File dataFileDirectory) {
    return getFile(dataFileDirectory, "testIdx");
  }

  public static File getTrackFile(File dataFileDirectory) {
    return getFile(dataFileDirectory, "trackData");
  }

  private static File getFile(File dataFileDirectory, String prefix) {
    // Works on set 1 or 2
    for (int set : new int[] {1,2}) {
      // Works on sample data from before contest or real data
      for (String firstLinesOrNot : new String[] {"", ".firstLines"}) {
        for (String gzippedOrNot : new String[] {".gz", ""}) {
          File dataFile = new File(dataFileDirectory, prefix + set + firstLinesOrNot + ".txt" + gzippedOrNot);
          if (dataFile.exists()) {
            return dataFile;
          }
        }
      }
    }
    throw new IllegalArgumentException("Can't find " + prefix + " file in " + dataFileDirectory);
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    return delegate.getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    return delegate.getPreferencesFromUser(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return delegate.getItemIDsFromUser(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    return delegate.getItemIDs();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return delegate.getPreferencesForItem(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceValue(userID, itemID);
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceTime(userID, itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
    return delegate.getNumItems();
  }

  @Override
  public int getNumUsers() throws TasteException {
    return delegate.getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemID);
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemID1, itemID2);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    delegate.setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    delegate.removePreference(userID, itemID);
  }

  @Override
  public boolean hasPreferenceValues() {
    return delegate.hasPreferenceValues();
  }

  @Override
  public float getMaxPreference() {
    return 100.0f;
  }

  @Override
  public float getMinPreference() {
    return 0.0f;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
