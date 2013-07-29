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

package org.apache.mahout.cf.taste.impl.model.hbase;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.KeyValue;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.HTableFactory;
import org.apache.hadoop.hbase.client.HTableInterface;
import org.apache.hadoop.hbase.client.HTablePool;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.filter.FilterList;
import org.apache.hadoop.hbase.filter.KeyOnlyFilter;
import org.apache.hadoop.hbase.filter.FirstKeyOnlyFilter;
import org.apache.hadoop.hbase.util.Bytes;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * <p>Naive approach of storing one preference as one value in the table.
 * Preferences are indexed as (user, item) and (item, user) for O(1) lookups.</p>
 *
 * <p>The default table name is "taste", this can be set through a constructor
 * argument. Each row has a value starting with "i" or "u" followed by the
 * actual id encoded as a big endian long.</p>
 *
 * <p>E.g., "u\x00\x00\x00\x00\x00\x00\x04\xd2" is user 1234L</p>
 *
 * <p>There are two column families: "users" and "items".</p>
 *
 * <p>The "users" column family holds user->item preferences. Each userID is the
 * column qualifier and the value is the preference.</p>
 *
 * <p>The "items" column fmaily holds item->user preferences. Each itemID is the
 * column qualifier and the value is the preference.</p>
 *
 * <p>User IDs and item IDs are cached in a FastIDSet since it requires a full
 * table scan to build these sets. Preferences are not cached since they
 * are pretty cheap lookups in HBase (also caching the Preferences defeats
 * the purpose of a scalable storage engine like HBase).</p>
 */
public final class HBaseDataModel implements DataModel, Closeable {

  private static final Logger log = LoggerFactory.getLogger(HBaseDataModel.class);

  private static final String DEFAULT_TABLE = "taste";
  private static final byte[] USERS_CF = Bytes.toBytes("users");
  private static final byte[] ITEMS_CF = Bytes.toBytes("items");

  private final HTablePool pool;
  private final String tableName;

  // Cache of user and item ids
  private volatile FastIDSet itemIDs;
  private volatile FastIDSet userIDs;

  public HBaseDataModel(String zkConnect) throws IOException {
    this(zkConnect, DEFAULT_TABLE);
  }

  public HBaseDataModel(String zkConnect, String tableName) throws IOException {
    log.info("Using HBase table {}", tableName);
    Configuration conf = HBaseConfiguration.create();
    conf.set("hbase.zookeeper.quorum", zkConnect);
    HTableFactory tableFactory = new HTableFactory();
    this.pool = new HTablePool(conf, 8, tableFactory);
    this.tableName = tableName;

    bootstrap(conf);
    // Warm the cache
    refresh(null);
  }

  public HBaseDataModel(HTablePool pool, String tableName, Configuration conf) throws IOException {
    log.info("Using HBase table {}", tableName);
    this.pool = pool;
    this.tableName = tableName;

    bootstrap(conf);

    // Warm the cache
    refresh(null);
  }

  public String getTableName() {
    return tableName;
  }

  /**
   * Create the table if it doesn't exist
   */
  private void bootstrap(Configuration conf) throws IOException {
    HBaseAdmin admin = new HBaseAdmin(conf);
    HTableDescriptor tDesc = new HTableDescriptor(Bytes.toBytes(tableName));
    tDesc.addFamily(new HColumnDescriptor(USERS_CF));
    tDesc.addFamily(new HColumnDescriptor(ITEMS_CF));
    try {
      admin.createTable(tDesc);
      log.info("Created table {}", tableName);
    } finally {
      admin.close();
    }
  }

  /**
   * Prefix a user id with "u" and convert to byte[]
   */
  private static byte[] userToBytes(long userID) {
    ByteBuffer bb = ByteBuffer.allocate(9);
    bb.put((byte)0x75); // The letter "u"
    bb.putLong(userID);
    return bb.array();
  }

  /**
   * Prefix an item id with "i" and convert to byte[]
   */
  private static byte[] itemToBytes(long itemID) {
    ByteBuffer bb = ByteBuffer.allocate(9);
    bb.put((byte)0x69); // The letter "i"
    bb.putLong(itemID);
    return bb.array();
  }

  /**
   * Extract the id out of a prefix byte[] id
   */
  private static long bytesToUserOrItemID(byte[] ba) {
    ByteBuffer bb = ByteBuffer.wrap(ba);
    return bb.getLong(1);
  }

  /* DataModel interface */

  @Override
  public LongPrimitiveIterator getUserIDs() {
    return userIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    Result result;
    try {
      HTableInterface table = pool.getTable(tableName);
      Get get = new Get(userToBytes(userID));
      get.addFamily(ITEMS_CF);
      result = table.get(get);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve user preferences from HBase", e);
    }

    if (result.isEmpty()) {
      throw new NoSuchUserException(userID);
    }

    SortedMap<byte[], byte[]> families = result.getFamilyMap(ITEMS_CF);
    PreferenceArray prefs = new GenericUserPreferenceArray(families.size());
    prefs.setUserID(0, userID);
    int i = 0;
    for (Map.Entry<byte[], byte[]> entry : families.entrySet()) {
      prefs.setItemID(i, Bytes.toLong(entry.getKey()));
      prefs.setValue(i, Bytes.toFloat(entry.getValue()));
      i++;
    }
    return prefs;
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    Result result;
    try {
      HTableInterface table = pool.getTable(tableName);
      Get get = new Get(userToBytes(userID));
      get.addFamily(ITEMS_CF);
      result = table.get(get);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve item IDs from HBase", e);
    }

    if (result.isEmpty()) {
      throw new NoSuchUserException(userID);
    }

    SortedMap<byte[],byte[]> families = result.getFamilyMap(ITEMS_CF);
    FastIDSet ids = new FastIDSet(families.size());
    for (byte[] family : families.keySet()) {
      ids.add(Bytes.toLong(family));
    }
    return ids;
  }

  @Override
  public LongPrimitiveIterator getItemIDs() {
    return itemIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    Result result;
    try {
      HTableInterface table = pool.getTable(tableName);
      Get get = new Get(itemToBytes(itemID));
      get.addFamily(USERS_CF);
      result = table.get(get);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve item preferences from HBase", e);
    }

    if (result.isEmpty()) {
      throw new NoSuchItemException(itemID);
    }

    SortedMap<byte[], byte[]> families = result.getFamilyMap(USERS_CF);
    PreferenceArray prefs = new GenericItemPreferenceArray(families.size());
    prefs.setItemID(0, itemID);
    int i = 0;
    for (Map.Entry<byte[], byte[]> entry : families.entrySet()) {
      prefs.setUserID(i, Bytes.toLong(entry.getKey()));
      prefs.setValue(i, Bytes.toFloat(entry.getValue()));
      i++;
    }
    return prefs;
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    Result result;
    try {
      HTableInterface table = pool.getTable(tableName);
      Get get = new Get(userToBytes(userID));
      get.addColumn(ITEMS_CF, Bytes.toBytes(itemID));
      result = table.get(get);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve user preferences from HBase", e);
    }

    if (result.isEmpty()) {
      throw new NoSuchUserException(userID);
    }

    if (result.containsColumn(ITEMS_CF, Bytes.toBytes(itemID))) {
      return Bytes.toFloat(result.getValue(ITEMS_CF, Bytes.toBytes(itemID)));
    } else {
      return null;
    }
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    Result result;
    try {
      HTableInterface table = pool.getTable(tableName);
      Get get = new Get(userToBytes(userID));
      get.addColumn(ITEMS_CF, Bytes.toBytes(itemID));
      result = table.get(get);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve user preferences from HBase", e);
    }

    if (result.isEmpty()) {
      throw new NoSuchUserException(userID);
    }

    if (result.containsColumn(ITEMS_CF, Bytes.toBytes(itemID))) {
      KeyValue kv = result.getColumnLatest(ITEMS_CF, Bytes.toBytes(itemID));
      return kv.getTimestamp();
    } else {
      return null;
    }
  }

  @Override
  public int getNumItems() {
    return itemIDs.size();
  }

  @Override
  public int getNumUsers() {
    return userIDs.size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    PreferenceArray prefs = getPreferencesForItem(itemID);
    return prefs.length();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    Result[] results;
    try {
      HTableInterface table = pool.getTable(tableName);
      List<Get> gets = Lists.newArrayListWithCapacity(2);
      gets.add(new Get(itemToBytes(itemID1)));
      gets.add(new Get(itemToBytes(itemID2)));
      gets.get(0).addFamily(USERS_CF);
      gets.get(1).addFamily(USERS_CF);
      results = table.get(gets);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to retrieve item preferences from HBase", e);
    }

    if (results[0].isEmpty()) {
      throw new NoSuchItemException(itemID1);
    }
    if (results[1].isEmpty()) {
      throw new NoSuchItemException(itemID2);
    }

    // First item
    Result result = results[0];
    SortedMap<byte[], byte[]> families = result.getFamilyMap(USERS_CF);
    FastIDSet idSet1 = new FastIDSet(families.size());
    for (byte[] id : families.keySet()) {
      idSet1.add(Bytes.toLong(id));
    }

    // Second item
    result = results[1];
    families = result.getFamilyMap(USERS_CF);
    FastIDSet idSet2 = new FastIDSet(families.size());
    for (byte[] id : families.keySet()) {
      idSet2.add(Bytes.toLong(id));
    }

    return idSet1.intersectionSize(idSet2);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    try {
      HTableInterface table = pool.getTable(tableName);
      List<Put> puts = Lists.newArrayListWithCapacity(2);
      puts.add(new Put(userToBytes(userID)));
      puts.add(new Put(itemToBytes(itemID)));
      puts.get(0).add(ITEMS_CF, Bytes.toBytes(itemID), Bytes.toBytes(value));
      puts.get(1).add(USERS_CF, Bytes.toBytes(userID), Bytes.toBytes(value));
      table.put(puts);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to store preference in HBase", e);
    }
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    try {
      HTableInterface table = pool.getTable(tableName);
      List<Delete> deletes = Lists.newArrayListWithCapacity(2);
      deletes.add(new Delete(userToBytes(userID)));
      deletes.add(new Delete(itemToBytes(itemID)));
      deletes.get(0).deleteColumns(ITEMS_CF, Bytes.toBytes(itemID));
      deletes.get(1).deleteColumns(USERS_CF, Bytes.toBytes(userID));
      table.delete(deletes);
      table.close();
    } catch (IOException e) {
      throw new TasteException("Failed to remove preference from HBase", e);
    }
  }

  @Override
  public boolean hasPreferenceValues() {
    return true;
  }

  @Override
  public float getMaxPreference() {
    throw new UnsupportedOperationException();
  }

  @Override
  public float getMinPreference() {
    throw new UnsupportedOperationException();
  }

  /* Closeable interface */

  @Override
  public void close() throws IOException {
    pool.close();
  }

  /* Refreshable interface */

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    if (alreadyRefreshed == null || !alreadyRefreshed.contains(this)) {
      try {
        log.info("Refreshing item and user ID caches");
        long t1 = System.currentTimeMillis();
        refreshItemIDs();
        refreshUserIDs();
        long t2 = System.currentTimeMillis();
        log.info("Finished refreshing caches in {} ms", t2 - t1);
      } catch (IOException e) {
        throw new IllegalStateException("Could not reload DataModel", e);
      }
    }
  }

  /*
   * Refresh the item id cache. Warning: this does a large table scan
   */
  private synchronized void refreshItemIDs() throws IOException {
    // Get the list of item ids
    HTableInterface table = pool.getTable(tableName);
    Scan scan = new Scan(new byte[]{0x69}, new byte[]{0x70});
    scan.setFilter(new FilterList(FilterList.Operator.MUST_PASS_ALL, new KeyOnlyFilter(), new FirstKeyOnlyFilter()));
    ResultScanner scanner = table.getScanner(scan);
    Collection<Long> ids = Lists.newLinkedList();
    for (Result result : scanner) {
      ids.add(bytesToUserOrItemID(result.getRow()));
    }
    table.close();

    // Copy into FastIDSet
    FastIDSet itemIDs = new FastIDSet(ids.size());
    for (long l : ids) {
      itemIDs.add(l);
    }

    // Swap with the active
    this.itemIDs = itemIDs;
  }

  /*
   * Refresh the user id cache. Warning: this does a large table scan
   */
  private synchronized void refreshUserIDs() throws IOException {
    // Get the list of user ids
    HTableInterface table = pool.getTable(tableName);
    Scan scan = new Scan(new byte[]{0x75}, new byte[]{0x76});
    scan.setFilter(new FilterList(FilterList.Operator.MUST_PASS_ALL, new KeyOnlyFilter(), new FirstKeyOnlyFilter()));
    ResultScanner scanner = table.getScanner(scan);
    Collection<Long> ids = Lists.newLinkedList();
    for (Result result : scanner) {
      ids.add(bytesToUserOrItemID(result.getRow()));
    }
    table.close();

    // Copy into FastIDSet
    FastIDSet userIDs = new FastIDSet(ids.size());
    for (long l : ids) {
      userIDs.add(l);
    }

    // Swap with the active
    this.userIDs = userIDs;
  }

}
