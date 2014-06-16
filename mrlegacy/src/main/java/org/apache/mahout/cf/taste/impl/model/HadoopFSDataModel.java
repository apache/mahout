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

package org.apache.mahout.cf.taste.impl.model;

import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is a child of FileDataModel,
 * and used to build model by reading data from hadoop filesystem
 */
public class HadoopFSDataModel extends FileDataModel {

  private static final Logger log = LoggerFactory.getLogger(HadoopFSDataModel.class);

  public static final long DEFAULT_MIN_RELOAD_INTERVAL_MS = 60 * 1000L; // 1 minute?
  
  private final FileSystem fs;
  private FileStatus fileStatus;

  /**
   * @param fs
   * hadoop filesystem, (e.g. hdfs)
   * @param file
   * the path of the file in filesystem      
   * @throws IOException
   */
  public HadoopFSDataModel(FileSystem fs, Path file) throws IOException {
    this(fs, file, false, DEFAULT_MIN_RELOAD_INTERVAL_MS);
  }
  
  public HadoopFSDataModel(FileSystem fs, Path file, String delimiterRegex) throws IOException {
    this(fs, file, false, DEFAULT_MIN_RELOAD_INTERVAL_MS, delimiterRegex);
  }
  
  public HadoopFSDataModel(FileSystem fs, Path file, boolean transpose, long minReloadIntervalMS) 
      throws IOException {
    this(fs, file, transpose, minReloadIntervalMS, null);
  }
  
  public HadoopFSDataModel(FileSystem fs, Path file, boolean transpose, 
      long minReloadIntervalMS, String delimiterRegex) throws IOException {
    log.info("Creating FileDataModel for inputstream");
    
    FileStatus filestat = fs.getFileStatus(file);
    this.fs = fs;
    this.fileStatus = filestat;
    this.lastModified = filestat.getModificationTime();
    this.lastUpdateFileModified = readLastUpdateModifiedFromFS();
    
    // read file from filesystem
    FSDataInputStream fsin = fs.open(file);
    FileLineIterator iterator = new FileLineIterator(fsin);
    
    initialize(iterator, transpose, minReloadIntervalMS, delimiterRegex);
  }
  
  public FileSystem getFileSystem() {
    return fs;
  }
  
  public FileStatus getFileStatus() {
    return fileStatus;
  }

  @Override
  protected DataModel buildModel() throws IOException {

    long newLastModified = fileStatus.getModificationTime();
    long newLastUpdateFileModified = readLastUpdateModifiedFromFS();

    boolean loadFreshData = delegate == null || newLastModified > lastModified + minReloadIntervalMS;

    long oldLastUpdateFileModifieid = lastUpdateFileModified;
    lastModified = newLastModified;
    lastUpdateFileModified = newLastUpdateFileModified;

    FastByIDMap<FastByIDMap<Long>> timestamps = new FastByIDMap<FastByIDMap<Long>>();

    if (hasPrefValues) {
      if (loadFreshData) {
        FastByIDMap<Collection<Preference>> data = new FastByIDMap<Collection<Preference>>();
        FSDataInputStream fsin = fs.open(fileStatus.getPath());
        FileLineIterator iterator = new FileLineIterator(fsin);
        processFile(iterator, data, timestamps, false);

        for (FileStatus stat : findUpdateFileStatusAfter(newLastModified)) {
          FSDataInputStream in = fs.open(stat.getPath());
          processFile(new FileLineIterator(in), data, timestamps, false);
        }

        return new GenericDataModel(GenericDataModel.toDataMap(data, true), timestamps);

      } else {
        FastByIDMap<PreferenceArray> rawData = ((GenericDataModel) delegate).getRawUserData();

        for (FileStatus stat : findUpdateFileStatusAfter(Math.max(oldLastUpdateFileModifieid, newLastModified))) {
          FSDataInputStream in = fs.open(stat.getPath());
          processFile(new FileLineIterator(in), rawData, timestamps, true);
        }

        return new GenericDataModel(rawData, timestamps);

      }

    } else {
      if (loadFreshData) {
        FastByIDMap<FastIDSet> data = new FastByIDMap<FastIDSet>();
        FSDataInputStream fsin = fs.open(fileStatus.getPath());
        FileLineIterator iterator = new FileLineIterator(fsin);
        processFileWithoutID(iterator, data, timestamps);

        for (FileStatus stat : findUpdateFileStatusAfter(newLastModified)) {
          FSDataInputStream in = fs.open(stat.getPath());
          processFileWithoutID(new FileLineIterator(in), data, timestamps);
        }

        return new GenericBooleanPrefDataModel(data, timestamps);

      } else {
        FastByIDMap<FastIDSet> rawData = ((GenericBooleanPrefDataModel) delegate).getRawUserData();

        for (FileStatus stat : findUpdateFileStatusAfter(Math.max(oldLastUpdateFileModifieid, newLastModified))) {
          FSDataInputStream in = fs.open(stat.getPath());
          processFileWithoutID(new FileLineIterator(in), rawData, timestamps);
        }

        return new GenericBooleanPrefDataModel(rawData, timestamps);
      }
    }
  }

  private Iterable<FileStatus> findUpdateFileStatusAfter(long minimumLastModified) throws IOException {
    Path path = fileStatus.getPath();
    String dataFileName = path.getName();
    int period = dataFileName.indexOf('.');
    String startName = period < 0 ? dataFileName : dataFileName.substring(0, period);
    Path parentPath = path.getParent();
    Map<Long, FileStatus> modTimeToUpdateFile = new TreeMap<Long,FileStatus>();
    for (FileStatus stat : fs.listStatus(parentPath)) {
      String updateFileName = stat.getPath().getName();
      if (updateFileName.startsWith(startName)
          && !updateFileName.equals(dataFileName)
          && stat.getModificationTime() >= minimumLastModified) {
        modTimeToUpdateFile.put(stat.getModificationTime(), stat);
      }
    }
    return modTimeToUpdateFile.values();
  }
  
  private long readLastUpdateModifiedFromFS() throws IOException {
    long mostRecentModification = Long.MIN_VALUE;
    for (FileStatus fStatus : findUpdateFileStatusAfter(0L)) {
      mostRecentModification = Math.max(mostRecentModification, fStatus.getModificationTime());
    }
    return mostRecentModification;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    try {
      FileStatus stat = fs.getFileStatus(fileStatus.getPath());
      if (stat.getModificationTime() != fileStatus.getModificationTime()) {
        fileStatus = stat;
      }
  
      if (fileStatus.getModificationTime() > lastModified + minReloadIntervalMS
          || readLastUpdateModifiedFromFS() > lastUpdateFileModified + minReloadIntervalMS) {
        log.debug("File has changed; reloading...");
        reload();
      }
    } catch (IOException e) {
      log.error(e.getMessage());
    }
  }

  @Override
  public String toString() {
    return "HadoopFSDataModel[dataFile:" + fileStatus.getPath().getName() + ']';
  }

}
