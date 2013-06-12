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

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.regex.Pattern;

import com.google.common.collect.AbstractIterator;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.impl.common.SkippingIterator;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>An {@link java.util.Iterator} which iterates over any of the KDD Cup's rating files. These include the files
 * {train,test,validation}Idx{1,2}}.txt. See http://kddcup.yahoo.com/. Each element in the iteration corresponds
 * to one user's ratings as a {@link PreferenceArray} and corresponding timestamps as a parallel {@code long}
 * array.</p>
 *
 * <p>Timestamps in the data set are relative to some unknown point in time, for anonymity. They are assumed
 * to be relative to the epoch, time 0, or January 1 1970, for purposes here.</p>
 */
public final class DataFileIterator
    extends AbstractIterator<Pair<PreferenceArray,long[]>>
    implements SkippingIterator<Pair<PreferenceArray,long[]>>, Closeable {

  private static final Pattern COLON_PATTERN = Pattern.compile(":");
  private static final Pattern PIPE_PATTERN = Pattern.compile("\\|");
  private static final Pattern TAB_PATTERN = Pattern.compile("\t");

  private final FileLineIterator lineIterator;

  private static final Logger log = LoggerFactory.getLogger(DataFileIterator.class);

  public DataFileIterator(File dataFile) throws IOException {
    if (dataFile == null || dataFile.isDirectory() || !dataFile.exists()) {
      throw new IllegalArgumentException("Bad data file: " + dataFile);
    }
    lineIterator = new FileLineIterator(dataFile);
  }

  @Override
  protected Pair<PreferenceArray, long[]> computeNext() {

    if (!lineIterator.hasNext()) {
      return endOfData();
    }

    String line = lineIterator.next();
    // First a userID|ratingsCount line
    String[] tokens = PIPE_PATTERN.split(line);

    long userID = Long.parseLong(tokens[0]);
    int ratingsLeftToRead = Integer.parseInt(tokens[1]);
    int ratingsRead = 0;

    PreferenceArray currentUserPrefs = new GenericUserPreferenceArray(ratingsLeftToRead);
    long[] timestamps = new long[ratingsLeftToRead];

    while (ratingsLeftToRead > 0) {

      line = lineIterator.next();

      // Then a data line. May be 1-4 tokens depending on whether preference info is included (it's not in test data)
      // or whether date info is included (not inluded in track 2). Item ID is always first, and date is the last
      // two fields if it exists.
      tokens = TAB_PATTERN.split(line);
      boolean hasPref = tokens.length == 2 || tokens.length == 4;
      boolean hasDate = tokens.length > 2;

      long itemID = Long.parseLong(tokens[0]);

      currentUserPrefs.setUserID(0, userID);
      currentUserPrefs.setItemID(ratingsRead, itemID);
      if (hasPref) {
        float preference = Float.parseFloat(tokens[1]);
        currentUserPrefs.setValue(ratingsRead, preference);
      }

      if (hasDate) {
        long timestamp;
        if (hasPref) {
          timestamp = parseFakeTimestamp(tokens[2], tokens[3]);
        } else {
          timestamp = parseFakeTimestamp(tokens[1], tokens[2]);
        }
        timestamps[ratingsRead] = timestamp;
      }

      ratingsRead++;
      ratingsLeftToRead--;
    }

    return new Pair<PreferenceArray,long[]>(currentUserPrefs, timestamps);
  }

  @Override
  public void skip(int n) {
    for (int i = 0; i < n; i++) {
      if (lineIterator.hasNext()) {
        String line = lineIterator.next();
        // First a userID|ratingsCount line
        String[] tokens = PIPE_PATTERN.split(line);
        int linesToSKip = Integer.parseInt(tokens[1]);
        lineIterator.skip(linesToSKip);
      } else {
        break;
      }
    }
  }

  @Override
  public void close() {
    endOfData();
    try {
      Closeables.close(lineIterator, true);
    } catch (IOException e) {
      log.error(e.getMessage(), e);
    }
  }

  /**
   * @param dateString "date" in days since some undisclosed date, which we will arbitrarily assume to be the
   *  epoch, January 1 1970.
   * @param timeString time of day in HH:mm:ss format
   * @return the UNIX timestamp for this moment in time
   */
  private static long parseFakeTimestamp(String dateString, CharSequence timeString) {
    int days = Integer.parseInt(dateString);
    String[] timeTokens = COLON_PATTERN.split(timeString);
    int hours = Integer.parseInt(timeTokens[0]);
    int minutes = Integer.parseInt(timeTokens[1]);
    int seconds = Integer.parseInt(timeTokens[2]);
    return 86400L * days + 3600L + hours + 60L * minutes + seconds;
  }

}
