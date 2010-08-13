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

package org.apache.mahout.cf.taste.impl.recommender.slopeone.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.InvertedRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;
import org.apache.mahout.common.FileLineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * {@link DiffStorage} which reads pre-computed diffs from a file and stores in memory. The file should have
 * one diff per line:
 * </p>
 * 
 * {@code itemID1,itemID2,diff}
 * 
 * <p>
 * Commas or tabs can be delimiters. This is intended for use in conjuction with the output of
 * {@link org.apache.mahout.cf.taste.hadoop.slopeone.SlopeOneAverageDiffsJob}.
 * </p>
 */
public final class FileDiffStorage implements DiffStorage {
  
  private static final Logger log = LoggerFactory.getLogger(FileDiffStorage.class);
  
  private static final long MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?
  private static final char COMMENT_CHAR = '#';
  
  private final File dataFile;
  private long lastModified;
  private boolean loaded;
  private final long maxEntries;
  private final FastByIDMap<FastByIDMap<RunningAverage>> averageDiffs;
  private final FastIDSet allRecommendableItemIDs;
  private final ReadWriteLock buildAverageDiffsLock;
  
  /**
   * @param dataFile
   *          diffs file
   * @param maxEntries
   *          maximum number of diffs to store
   * @throws FileNotFoundException
   *           if data file does not exist or is a directory
   */
  public FileDiffStorage(File dataFile, long maxEntries) throws FileNotFoundException {
    if (dataFile == null) {
      throw new IllegalArgumentException("dataFile is null");
    }
    if (!dataFile.exists() || dataFile.isDirectory()) {
      throw new FileNotFoundException(dataFile.toString());
    }
    if (maxEntries <= 0L) {
      throw new IllegalArgumentException("maxEntries must be positive");
    }
    
    log.info("Creating FileDataModel for file {}", dataFile);
    
    this.dataFile = dataFile.getAbsoluteFile();
    this.lastModified = dataFile.lastModified();
    this.maxEntries = maxEntries;
    this.averageDiffs = new FastByIDMap<FastByIDMap<RunningAverage>>();
    this.allRecommendableItemIDs = new FastIDSet();
    this.buildAverageDiffsLock = new ReentrantReadWriteLock();
  }
  
  private void buildDiffs() {
    if (buildAverageDiffsLock.writeLock().tryLock()) {
      try {
        
        averageDiffs.clear();
        allRecommendableItemIDs.clear();
        
        FileLineIterator iterator = new FileLineIterator(dataFile, false);
        String firstLine = iterator.peek();
        while (firstLine.length() == 0 || firstLine.charAt(0) == COMMENT_CHAR) {
          iterator.next();
          firstLine = iterator.peek();
        }
        char delimiter = FileDataModel.determineDelimiter(firstLine);
        long averageCount = 0L;
        while (iterator.hasNext()) {
          averageCount = processLine(iterator.next(), delimiter, averageCount);
        }
        
        pruneInconsequentialDiffs();
        updateAllRecommendableItems();
        
      } catch (IOException ioe) {
        log.warn("Exception while reloading", ioe);
      } finally {
        buildAverageDiffsLock.writeLock().unlock();
      }
    }
  }
  
  private long processLine(String line, char delimiter, long averageCount) {
    
    if ((line.length() == 0) || (line.charAt(0) == COMMENT_CHAR)) {
      return averageCount;
    }
    
    int delimiterOne = line.indexOf(delimiter);
    if (delimiterOne < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }
    int delimiterTwo = line.indexOf(delimiter, delimiterOne + 1);
    if (delimiterTwo < 0) {
      throw new IllegalArgumentException("Bad line: " + line);
    }
    
    long itemID1 = Long.parseLong(line.substring(0, delimiterOne));
    long itemID2 = Long.parseLong(line.substring(delimiterOne + 1, delimiterTwo));
    double diff = Double.parseDouble(line.substring(delimiterTwo + 1));
    
    if (itemID1 > itemID2) {
      long temp = itemID1;
      itemID1 = itemID2;
      itemID2 = temp;
    }
    
    FastByIDMap<RunningAverage> level1Map = averageDiffs.get(itemID1);
    if (level1Map == null) {
      level1Map = new FastByIDMap<RunningAverage>();
      averageDiffs.put(itemID1, level1Map);
    }
    RunningAverage average = level1Map.get(itemID2);
    if ((average == null) && (averageCount < maxEntries)) {
      average = new FullRunningAverage();
      level1Map.put(itemID2, average);
      averageCount++;
    }
    if (average != null) {
      average.addDatum(diff);
    }
    
    allRecommendableItemIDs.add(itemID1);
    allRecommendableItemIDs.add(itemID2);
    
    return averageCount;
  }
  
  private void pruneInconsequentialDiffs() {
    // Go back and prune inconsequential diffs. "Inconsequential" means, here, only represented by one
    // data point, so possibly unreliable
    Iterator<Map.Entry<Long,FastByIDMap<RunningAverage>>> it1 = averageDiffs.entrySet().iterator();
    while (it1.hasNext()) {
      FastByIDMap<RunningAverage> map = it1.next().getValue();
      Iterator<Map.Entry<Long,RunningAverage>> it2 = map.entrySet().iterator();
      while (it2.hasNext()) {
        RunningAverage average = it2.next().getValue();
        if (average.getCount() <= 1) {
          it2.remove();
        }
      }
      if (map.isEmpty()) {
        it1.remove();
      } else {
        map.rehash();
      }
    }
    averageDiffs.rehash();
  }
  
  private void updateAllRecommendableItems() {
    for (Map.Entry<Long,FastByIDMap<RunningAverage>> entry : averageDiffs.entrySet()) {
      allRecommendableItemIDs.add(entry.getKey());
      LongPrimitiveIterator it = entry.getValue().keySetIterator();
      while (it.hasNext()) {
        allRecommendableItemIDs.add(it.next());
      }
    }
    allRecommendableItemIDs.rehash();
  }
  
  private void checkLoaded() {
    if (!loaded) {
      buildDiffs();
      loaded = true;
    }
  }
  
  @Override
  public RunningAverage getDiff(long itemID1, long itemID2) {
    checkLoaded();
    
    boolean inverted = false;
    if (itemID1 > itemID2) {
      inverted = true;
      long temp = itemID1;
      itemID1 = itemID2;
      itemID2 = temp;
    }
    
    FastByIDMap<RunningAverage> level2Map;
    try {
      buildAverageDiffsLock.readLock().lock();
      level2Map = averageDiffs.get(itemID1);
    } finally {
      buildAverageDiffsLock.readLock().unlock();
    }
    RunningAverage average = null;
    if (level2Map != null) {
      average = level2Map.get(itemID2);
    }
    if (inverted) {
      if (average == null) {
        return null;
      }
      return new InvertedRunningAverage(average);
    } else {
      return average;
    }
  }
  
  @Override
  public RunningAverage[] getDiffs(long userID, long itemID, PreferenceArray prefs) {
    checkLoaded();
    try {
      buildAverageDiffsLock.readLock().lock();
      int size = prefs.length();
      RunningAverage[] result = new RunningAverage[size];
      for (int i = 0; i < size; i++) {
        result[i] = getDiff(prefs.getItemID(i), itemID);
      }
      return result;
    } finally {
      buildAverageDiffsLock.readLock().unlock();
    }
  }
  
  @Override
  public RunningAverage getAverageItemPref(long itemID) {
    checkLoaded();
    return null; // TODO can't do this without a DataModel
  }
  
  @Override
  public void updateItemPref(long itemID, float prefDelta, boolean remove) {
    checkLoaded();
    try {
      buildAverageDiffsLock.readLock().lock();
      for (Map.Entry<Long,FastByIDMap<RunningAverage>> entry : averageDiffs.entrySet()) {
        boolean matchesItemID1 = itemID == entry.getKey();
        for (Map.Entry<Long,RunningAverage> entry2 : entry.getValue().entrySet()) {
          RunningAverage average = entry2.getValue();
          if (matchesItemID1) {
            if (remove) {
              average.removeDatum(prefDelta);
            } else {
              average.changeDatum(-prefDelta);
            }
          } else if (itemID == entry2.getKey()) {
            if (remove) {
              average.removeDatum(-prefDelta);
            } else {
              average.changeDatum(prefDelta);
            }
          }
        }
      }
      // RunningAverage itemAverage = averageItemPref.get(itemID);
      // if (itemAverage != null) {
      // itemAverage.changeDatum(prefDelta);
      // }
    } finally {
      buildAverageDiffsLock.readLock().unlock();
    }
  }
  
  @Override
  public FastIDSet getRecommendableItemIDs(long userID) {
    checkLoaded();
    try {
      buildAverageDiffsLock.readLock().lock();
      return allRecommendableItemIDs.clone();
    } finally {
      buildAverageDiffsLock.readLock().unlock();
    }
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    long mostRecentModification = dataFile.lastModified();
    if (mostRecentModification > lastModified + MIN_RELOAD_INTERVAL_MS) {
      log.debug("File has changed; reloading...");
      lastModified = mostRecentModification;
      buildDiffs();
    }
  }
  
}
