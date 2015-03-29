/*
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

package org.apache.mahout.cf.taste.impl.similarity.file;

import java.io.File;
import java.util.Collection;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * An {@link ItemSimilarity} backed by a comma-delimited file. This class typically expects a file where each line
 * contains an item ID, followed by another item ID, followed by a similarity value, separated by commas. You may also
 * use tabs.
 * </p>
 *
 * <p>
 * The similarity value is assumed to be parseable as a {@code double} having a value between -1 and 1. The
 * item IDs are parsed as {@code long}s. Similarities are symmetric so for a pair of items you do not have to
 * include 2 lines in the file.
 * </p>
 *
 * <p>
 * This class will reload data from the data file when {@link #refresh(Collection)} is called, unless the file
 * has been reloaded very recently already.
 * </p>
 *
 * <p>
 * This class is not intended for use with very large amounts of data. For that, a JDBC-backed {@link ItemSimilarity}
 * and a database are more appropriate.
 * </p>
 */
public class FileItemSimilarity implements ItemSimilarity {

  public static final long DEFAULT_MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?

  private ItemSimilarity delegate;
  private final ReentrantLock reloadLock;
  private final File dataFile;
  private long lastModified;
  private final long minReloadIntervalMS;

  private static final Logger log = LoggerFactory.getLogger(FileItemSimilarity.class);

  /**
   * @param dataFile
   *          file containing the similarity data
   */
  public FileItemSimilarity(File dataFile) {
    this(dataFile, DEFAULT_MIN_RELOAD_INTERVAL_MS);
  }

  /**
   * @param minReloadIntervalMS
   *          the minimum interval in milliseconds after which a full reload of the original datafile is done
   *          when refresh() is called
   * @see #FileItemSimilarity(File)
   */
  public FileItemSimilarity(File dataFile, long minReloadIntervalMS) {
    Preconditions.checkArgument(dataFile != null, "dataFile is null");
    Preconditions.checkArgument(dataFile.exists() && !dataFile.isDirectory(),
      "dataFile is missing or a directory: %s", dataFile);

    log.info("Creating FileItemSimilarity for file {}", dataFile);

    this.dataFile = dataFile.getAbsoluteFile();
    this.lastModified = dataFile.lastModified();
    this.minReloadIntervalMS = minReloadIntervalMS;
    this.reloadLock = new ReentrantLock();

    reload();
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    return delegate.itemSimilarities(itemID1, itemID2s);
  }

  @Override
  public long[] allSimilarItemIDs(long itemID) throws TasteException {
    return delegate.allSimilarItemIDs(itemID);
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    return delegate.itemSimilarity(itemID1, itemID2);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    if (dataFile.lastModified() > lastModified + minReloadIntervalMS) {
      log.debug("File has changed; reloading...");
      reload();
    }
  }

  protected void reload() {
    if (reloadLock.tryLock()) {
      try {
        long newLastModified = dataFile.lastModified();
        delegate = new GenericItemSimilarity(new FileItemItemSimilarityIterable(dataFile));
        lastModified = newLastModified;
      } finally {
        reloadLock.unlock();
      }
    }
  }

  @Override
  public String toString() {
    return "FileItemSimilarity[dataFile:" + dataFile + ']';
  }

}
