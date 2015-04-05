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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads user ids and item ids from files specified in usersFile, itemsFile or userItemFile options in item-based
 *  recommender. Composes a list of users and a list of items which can be used by
 * {@link org.apache.mahout.cf.taste.hadoop.item.UserVectorSplitterMapper} and
 * {@link org.apache.mahout.cf.taste.hadoop.item.AggregateAndRecommendReducer}.
 */
public class IDReader {

  static final String USER_ITEM_FILE = "userItemFile";

  private static final Logger log = LoggerFactory.getLogger(IDReader.class);
  private static final Pattern SEPARATOR = Pattern.compile("[\t,]");

  private Configuration conf;

  private String usersFile;
  private String itemsFile;
  private String userItemFile;

  private FastIDSet userIds;
  private FastIDSet itemIds;

  private FastIDSet emptySet;

  /* Key - user id, value - a set of item ids to include into recommendations for this user */
  private Map<Long, FastIDSet> userItemFilter;

  /**
   * Creates a new IDReader
   * 
   * @param conf Job configuration
   */
  public IDReader(Configuration conf) {
    this.conf = conf;
    emptySet = new FastIDSet();

    usersFile = conf.get(UserVectorSplitterMapper.USERS_FILE);
    itemsFile = conf.get(AggregateAndRecommendReducer.ITEMS_FILE);
    userItemFile = conf.get(USER_ITEM_FILE);
  }

  /**
   * Reads user ids and item ids from files specified in a job configuration
   * 
   * @throws IOException if an error occurs during file read operation
   * 
   * @throws IllegalStateException if userItemFile option is specified together with usersFile or itemsFile
   */
  public void readIDs() throws IOException, IllegalStateException {
    if (isUserItemFileSpecified()) {
      readUserItemFilterIfNeeded();
    }

    if (isUsersFileSpecified() || isUserItemFilterSpecified()) {
      readUserIds();
    }

    if (isItemsFileSpecified() || isUserItemFilterSpecified()) {
      readItemIds();
    }
  }

  /**
   * Gets a collection of items which should be recommended for a user
   * 
   * @param userId ID of a user we are interested in
   * @return if a userItemFile option is specified, and that file contains at least one item ID for the user,
   *         then this method returns a {@link FastIDSet} object populated with item IDs. Otherwise, this
   *         method returns an empty set.
   */
  public FastIDSet getItemsToRecommendForUser(Long userId) {
    if (isUserItemFilterSpecified() && userItemFilter.containsKey(userId)) {
      return userItemFilter.get(userId);
    } else {
      return emptySet;
    }
  }

  private void readUserIds() throws IOException, IllegalStateException {
    if (isUsersFileSpecified() && !isUserItemFileSpecified()) {
      userIds = readIDList(usersFile);
    } else if (isUserItemFileSpecified() && !isUsersFileSpecified()) {
      readUserItemFilterIfNeeded();
      userIds = extractAllUserIdsFromUserItemFilter(userItemFilter);
    } else if (!isUsersFileSpecified()) {
      throw new IllegalStateException("Neither usersFile nor userItemFile options are specified");
    } else {
      throw new IllegalStateException("usersFile and userItemFile options cannot be used simultaneously");
    }
  }

  private void readItemIds() throws IOException, IllegalStateException {
    if (isItemsFileSpecified() && !isUserItemFileSpecified()) {
      itemIds = readIDList(itemsFile);
    } else if (isUserItemFileSpecified() && !isItemsFileSpecified()) {
      readUserItemFilterIfNeeded();
      itemIds = extractAllItemIdsFromUserItemFilter(userItemFilter);
    } else if (!isItemsFileSpecified()) {
      throw new IllegalStateException("Neither itemsFile nor userItemFile options are specified");
    } else {
      throw new IllegalStateException("itemsFile and userItemFile options cannot be specified simultaneously");
    }
  }

  private void readUserItemFilterIfNeeded() throws IOException {
    if (!isUserItemFilterSpecified() && isUserItemFileSpecified()) {
      userItemFilter = readUserItemFilter(userItemFile);
    }
  }

  private Map<Long, FastIDSet> readUserItemFilter(String pathString) throws IOException {
    Map<Long, FastIDSet> result = new HashMap<>();

    try (InputStream in = openFile(pathString)) {
      for (String line : new FileLineIterable(in)) {
        try {
          String[] tokens = SEPARATOR.split(line);
          Long userId = Long.parseLong(tokens[0]);
          Long itemId = Long.parseLong(tokens[1]);

          addUserAndItemIdToUserItemFilter(result, userId, itemId);
        } catch (NumberFormatException nfe) {
          log.warn("userItemFile line ignored: {}", line);
        }
      }
    }

    return result;
  }

  void addUserAndItemIdToUserItemFilter(Map<Long, FastIDSet> filter, Long userId, Long itemId) {
    FastIDSet itemIds;

    if (filter.containsKey(userId)) {
      itemIds = filter.get(userId);
    } else {
      itemIds = new FastIDSet();
      filter.put(userId, itemIds);
    }

    itemIds.add(itemId);
  }

  static FastIDSet extractAllUserIdsFromUserItemFilter(Map<Long, FastIDSet> filter) {
    FastIDSet result = new FastIDSet();

    for (Long userId : filter.keySet()) {
      result.add(userId);
    }

    return result;
  }

  private FastIDSet extractAllItemIdsFromUserItemFilter(Map<Long, FastIDSet> filter) {
    FastIDSet result = new FastIDSet();

    for (FastIDSet itemIds : filter.values()) {
      result.addAll(itemIds);
    }

    return result;
  }

  private FastIDSet readIDList(String pathString) throws IOException {
    FastIDSet result = null;

    if (pathString != null) {
      result = new FastIDSet();

      try (InputStream in = openFile(pathString)){
        for (String line : new FileLineIterable(in)) {
          try {
            result.add(Long.parseLong(line));
          } catch (NumberFormatException nfe) {
            log.warn("line ignored: {}", line);
          }
        }
      }
    }

    return result;
  }

  private InputStream openFile(String pathString) throws IOException {
    return HadoopUtil.openStream(new Path(pathString), conf);
  }

  public boolean isUsersFileSpecified () {
    return usersFile != null;
  }
  
  public boolean isItemsFileSpecified () {
    return itemsFile != null;
  }
  
  public boolean isUserItemFileSpecified () {
    return userItemFile != null;
  }

  public boolean isUserItemFilterSpecified() {
    return userItemFilter != null;
  }

  public FastIDSet getUserIds() {
    return userIds;
  }

  public FastIDSet getItemIds() {
    return itemIds;
  }
}
