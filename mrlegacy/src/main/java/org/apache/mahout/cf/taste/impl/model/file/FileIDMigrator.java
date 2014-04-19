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

package org.apache.mahout.cf.taste.impl.model.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.locks.ReentrantLock;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.AbstractIDMigrator;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * An {@link org.apache.mahout.cf.taste.model.IDMigrator} backed by a file.
 * This class typically expects a file where each line
 * contains a single stringID to be stored in this migrator.
 * </p>
 *
 * <p>
 * This class will reload data from the data file when {@link #refresh(Collection)} is called, unless the file
 * has been reloaded very recently already.
 * </p>
 */
public class FileIDMigrator extends AbstractIDMigrator {

  public static final long DEFAULT_MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?

  private final File dataFile;
  private FastByIDMap<String> longToString;
  private final ReentrantLock reloadLock;

  private long lastModified;
  private final long minReloadIntervalMS;

  private static final Logger log = LoggerFactory.getLogger(FileIDMigrator.class);

  public FileIDMigrator(File dataFile) throws FileNotFoundException {
    this(dataFile, DEFAULT_MIN_RELOAD_INTERVAL_MS);
  }

  public FileIDMigrator(File dataFile, long minReloadIntervalMS) throws FileNotFoundException {
    longToString = new FastByIDMap<String>(100);
    this.dataFile = Preconditions.checkNotNull(dataFile);
    if (!dataFile.exists() || dataFile.isDirectory()) {
      throw new FileNotFoundException(dataFile.toString());
    }

    log.info("Creating FileReadonlyIDMigrator for file {}", dataFile);

    this.reloadLock = new ReentrantLock();
    this.lastModified = dataFile.lastModified();
    this.minReloadIntervalMS = minReloadIntervalMS;

    reload();
  }

  @Override
  public String toStringID(long longID) {
    return longToString.get(longID);
  }

  private void reload() {
    if (reloadLock.tryLock()) {
      try {
        longToString = buildMapping();
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      } finally {
        reloadLock.unlock();
      }
    }
  }

  private FastByIDMap<String> buildMapping() throws IOException {
    FastByIDMap<String> mapping = new FastByIDMap<String>();
    for (String line : new FileLineIterable(dataFile)) {
      mapping.put(toLongID(line), line);
    }
    lastModified = dataFile.lastModified();
    return mapping;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    if (dataFile.lastModified() > lastModified + minReloadIntervalMS) {
      log.debug("File has changed; reloading...");
      reload();
    }
  }

  @Override
  public String toString() {
    return "FileIDMigrator[dataFile:" + dataFile + ']';
  }
}
