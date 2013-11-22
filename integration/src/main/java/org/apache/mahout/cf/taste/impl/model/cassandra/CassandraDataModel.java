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

package org.apache.mahout.cf.taste.impl.model.cassandra;

import com.google.common.base.Preconditions;
import me.prettyprint.cassandra.model.HColumnImpl;
import me.prettyprint.cassandra.serializers.BytesArraySerializer;
import me.prettyprint.cassandra.serializers.FloatSerializer;
import me.prettyprint.cassandra.serializers.LongSerializer;
import me.prettyprint.cassandra.service.OperationType;
import me.prettyprint.hector.api.Cluster;
import me.prettyprint.hector.api.ConsistencyLevelPolicy;
import me.prettyprint.hector.api.HConsistencyLevel;
import me.prettyprint.hector.api.Keyspace;
import me.prettyprint.hector.api.beans.ColumnSlice;
import me.prettyprint.hector.api.beans.HColumn;
import me.prettyprint.hector.api.factory.HFactory;
import me.prettyprint.hector.api.mutation.Mutator;
import me.prettyprint.hector.api.query.ColumnQuery;
import me.prettyprint.hector.api.query.CountQuery;
import me.prettyprint.hector.api.query.SliceQuery;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import java.io.Closeable;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * <p>A {@link DataModel} based on a Cassandra keyspace. By default it uses keyspace "recommender" but this
 * can be configured. Create the keyspace before using this class; this can be done on the Cassandra command
 * line with a command linke {@code create keyspace recommender;}.</p>
 *
 * <p>Within the keyspace, this model uses four column families:</p>
 *
 * <p>First, it uses a column family called "users". This is keyed by the user ID as an 8-byte long.
 * It contains a column for every preference the user expresses. The column name is item ID, again as
 * an 8-byte long, and value is a floating point value represnted as an IEEE 32-bit floating poitn value.</p>
 *
 * <p>It uses an analogous column family called "items" for the same data, but keyed by item ID rather
 * than user ID. In this column family, column names are user IDs instead.</p>
 *
 * <p>It uses a column family called "userIDs" as well, with an identical schema. It has one row under key
 * 0. IT contains a column for every user ID in th emodel. It has no values.</p>
 *
 * <p>Finally it also uses an analogous column family "itemIDs" containing item IDs.</p>
 *
 * <p>Each of these four column families needs to be created ahead of time. Again the
 * Cassandra CLI can be used to do so, with commands like {@code create column family users;}.</p>
 *
 * <p>Note that this thread uses a long-lived Cassandra client which will run until terminated. You
 * must {@link #close()} this implementation when done or the JVM will not terminate.</p>
 *
 * <p>This implementation still relies heavily on reading data into memory and caching,
 * as it remains too data-intensive to be effective even against Cassandra. It will take some time to
 * "warm up" as the first few requests will block loading user and item data into caches. This is still going
 * to send a great deal of query traffic to Cassandra. It would be advisable to employ caching wrapper
 * classes in your implementation, like {@link org.apache.mahout.cf.taste.impl.recommender.CachingRecommender}
 * or {@link org.apache.mahout.cf.taste.impl.similarity.CachingItemSimilarity}.</p>
 */
public final class CassandraDataModel implements DataModel, Closeable {

  /** Default Cassandra host. Default: localhost */
  private static final String DEFAULT_HOST = "localhost";

  /** Default Cassandra port. Default: 9160 */
  private static final int DEFAULT_PORT = 9160;

  /** Default Cassandra keyspace. Default: recommender */
  private static final String DEFAULT_KEYSPACE = "recommender";

  static final String USERS_CF = "users";
  static final String ITEMS_CF = "items";
  static final String USER_IDS_CF = "userIDs";
  static final String ITEM_IDS_CF = "itemIDs";
  private static final long ID_ROW_KEY = 0L;
  private static final byte[] EMPTY = new byte[0];

  private final Cluster cluster;
  private final Keyspace keyspace;
  private final Cache<Long,PreferenceArray> userCache;
  private final Cache<Long,PreferenceArray> itemCache;
  private final Cache<Long,FastIDSet> itemIDsFromUserCache;
  private final Cache<Long,FastIDSet> userIDsFromItemCache;
  private final AtomicReference<Integer> userCountCache;
  private final AtomicReference<Integer> itemCountCache;

  /**
   * Uses the standard Cassandra host and port (localhost:9160), and keyspace name ("recommender").
   */
  public CassandraDataModel() {
    this(DEFAULT_HOST, DEFAULT_PORT, DEFAULT_KEYSPACE);
  }

  /**
   * @param host Cassandra server host name
   * @param port Cassandra server port
   * @param keyspaceName name of Cassandra keyspace to use
   */
  public CassandraDataModel(String host, int port, String keyspaceName) {
    
    Preconditions.checkNotNull(host);
    Preconditions.checkArgument(port > 0, "port must be greater then 0!");
    Preconditions.checkNotNull(keyspaceName);

    cluster = HFactory.getOrCreateCluster(CassandraDataModel.class.getSimpleName(), host + ':' + port);
    keyspace = HFactory.createKeyspace(keyspaceName, cluster);
    keyspace.setConsistencyLevelPolicy(new OneConsistencyLevelPolicy());

    userCache = new Cache<Long,PreferenceArray>(new UserPrefArrayRetriever(), 1 << 20);
    itemCache = new Cache<Long,PreferenceArray>(new ItemPrefArrayRetriever(), 1 << 20);
    itemIDsFromUserCache = new Cache<Long,FastIDSet>(new ItemIDsFromUserRetriever(), 1 << 20);
    userIDsFromItemCache = new Cache<Long,FastIDSet>(new UserIDsFromItemRetriever(), 1 << 20);
    userCountCache = new AtomicReference<Integer>(null);
    itemCountCache = new AtomicReference<Integer>(null);
  }

  @Override
  public LongPrimitiveIterator getUserIDs() {
    SliceQuery<Long,Long,?> query = buildNoValueSliceQuery(USER_IDS_CF);
    query.setKey(ID_ROW_KEY);
    FastIDSet userIDs = new FastIDSet();
    for (HColumn<Long,?> userIDColumn : query.execute().get().getColumns()) {
      userIDs.add(userIDColumn.getName());
    }
    return userIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    return userCache.get(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return itemIDsFromUserCache.get(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() {
    SliceQuery<Long,Long,?> query = buildNoValueSliceQuery(ITEM_IDS_CF);
    query.setKey(ID_ROW_KEY);
    FastIDSet itemIDs = new FastIDSet();
    for (HColumn<Long,?> itemIDColumn : query.execute().get().getColumns()) {
      itemIDs.add(itemIDColumn.getName());
    }
    return itemIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return itemCache.get(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) {
    ColumnQuery<Long,Long,Float> query =
        HFactory.createColumnQuery(keyspace, LongSerializer.get(), LongSerializer.get(), FloatSerializer.get());
    query.setColumnFamily(USERS_CF);
    query.setKey(userID);
    query.setName(itemID);
    HColumn<Long,Float> column = query.execute().get();
    return column == null ? null : column.getValue();
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) {
    ColumnQuery<Long,Long,?> query =
        HFactory.createColumnQuery(keyspace, LongSerializer.get(), LongSerializer.get(), BytesArraySerializer.get());
    query.setColumnFamily(USERS_CF);
    query.setKey(userID);
    query.setName(itemID);
    HColumn<Long,?> result = query.execute().get();
    return result == null ? null : result.getClock();
  }

  @Override
  public int getNumItems() {
    Integer itemCount = itemCountCache.get();
    if (itemCount == null) {
      CountQuery<Long,Long> countQuery =
          HFactory.createCountQuery(keyspace, LongSerializer.get(), LongSerializer.get());
      countQuery.setKey(ID_ROW_KEY);
      countQuery.setColumnFamily(ITEM_IDS_CF);
      countQuery.setRange(null, null, Integer.MAX_VALUE);
      itemCount = countQuery.execute().get();
      itemCountCache.set(itemCount);
    }
    return itemCount;
  }

  @Override
  public int getNumUsers() {
    Integer userCount = userCountCache.get();
    if (userCount == null) {
      CountQuery<Long,Long> countQuery =
          HFactory.createCountQuery(keyspace, LongSerializer.get(), LongSerializer.get());
      countQuery.setKey(ID_ROW_KEY);
      countQuery.setColumnFamily(USER_IDS_CF);
      countQuery.setRange(null, null, Integer.MAX_VALUE);
      userCount = countQuery.execute().get();
      userCountCache.set(userCount);
    }
    return userCount;
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    /*
    CountQuery<Long,Long> query = HFactory.createCountQuery(keyspace, LongSerializer.get(), LongSerializer.get());
    query.setColumnFamily(ITEMS_CF);
    query.setKey(itemID);
    query.setRange(null, null, Integer.MAX_VALUE);
    return query.execute().get();
     */
    return userIDsFromItemCache.get(itemID).size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    FastIDSet userIDs1 = userIDsFromItemCache.get(itemID1);
    FastIDSet userIDs2 = userIDsFromItemCache.get(itemID2);
    return userIDs1.size() < userIDs2.size()
        ? userIDs2.intersectionSize(userIDs1)
        : userIDs1.intersectionSize(userIDs2);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) {

    if (Float.isNaN(value)) {
      value = 1.0f;
    }
    
    long now = System.currentTimeMillis();

    Mutator<Long> mutator = HFactory.createMutator(keyspace, LongSerializer.get());

    HColumn<Long,Float> itemForUsers = new HColumnImpl<Long,Float>(LongSerializer.get(), FloatSerializer.get());
    itemForUsers.setName(itemID);
    itemForUsers.setClock(now);
    itemForUsers.setValue(value);
    mutator.addInsertion(userID, USERS_CF, itemForUsers);

    HColumn<Long,Float> userForItems = new HColumnImpl<Long,Float>(LongSerializer.get(), FloatSerializer.get());
    userForItems.setName(userID);
    userForItems.setClock(now);
    userForItems.setValue(value);
    mutator.addInsertion(itemID, ITEMS_CF, userForItems);

    HColumn<Long,byte[]> userIDs = new HColumnImpl<Long,byte[]>(LongSerializer.get(), BytesArraySerializer.get());
    userIDs.setName(userID);
    userIDs.setClock(now);
    userIDs.setValue(EMPTY);
    mutator.addInsertion(ID_ROW_KEY, USER_IDS_CF, userIDs);

    HColumn<Long,byte[]> itemIDs = new HColumnImpl<Long,byte[]>(LongSerializer.get(), BytesArraySerializer.get());
    itemIDs.setName(itemID);
    itemIDs.setClock(now);
    itemIDs.setValue(EMPTY);
    mutator.addInsertion(ID_ROW_KEY, ITEM_IDS_CF, itemIDs);

    mutator.execute();
  }

  @Override
  public void removePreference(long userID, long itemID) {
    Mutator<Long> mutator = HFactory.createMutator(keyspace, LongSerializer.get());
    mutator.addDeletion(userID, USERS_CF, itemID, LongSerializer.get());
    mutator.addDeletion(itemID, ITEMS_CF, userID, LongSerializer.get());
    mutator.execute();
    // Not deleting from userIDs, itemIDs though
  }

  /**
   * @return true
   */
  @Override
  public boolean hasPreferenceValues() {
    return true;
  }

  /**
   * @return Float#NaN
   */
  @Override
  public float getMaxPreference() {
    return Float.NaN;
  }

  /**
   * @return Float#NaN
   */
  @Override
  public float getMinPreference() {
    return Float.NaN;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    userCache.clear();
    itemCache.clear();
    userIDsFromItemCache.clear();
    itemIDsFromUserCache.clear();
    userCountCache.set(null);
    itemCountCache.set(null);
  }

  @Override
  public String toString() {
    return "CassandraDataModel[" + keyspace + ']';
  }

  @Override
  public void close() {
    HFactory.shutdownCluster(cluster);
  }


  private SliceQuery<Long,Long,byte[]> buildNoValueSliceQuery(String cf) {
    SliceQuery<Long,Long,byte[]> query =
        HFactory.createSliceQuery(keyspace, LongSerializer.get(), LongSerializer.get(), BytesArraySerializer.get());
    query.setColumnFamily(cf);
    query.setRange(null, null, false, Integer.MAX_VALUE);
    return query;
  }

  private SliceQuery<Long,Long,Float> buildValueSliceQuery(String cf) {
    SliceQuery<Long,Long,Float> query =
        HFactory.createSliceQuery(keyspace, LongSerializer.get(), LongSerializer.get(), FloatSerializer.get());
    query.setColumnFamily(cf);
    query.setRange(null, null, false, Integer.MAX_VALUE);
    return query;
  }


  private static final class OneConsistencyLevelPolicy implements ConsistencyLevelPolicy {
    @Override
    public HConsistencyLevel get(OperationType op) {
      return HConsistencyLevel.ONE;
    }

    @Override
    public HConsistencyLevel get(OperationType op, String cfName) {
      return HConsistencyLevel.ONE;
    }
  }

  private final class UserPrefArrayRetriever implements Retriever<Long, PreferenceArray> {
    @Override
    public PreferenceArray get(Long userID) throws TasteException {
      SliceQuery<Long,Long,Float> query = buildValueSliceQuery(USERS_CF);
      query.setKey(userID);

      ColumnSlice<Long,Float> result = query.execute().get();
      if (result == null) {
        throw new NoSuchUserException(userID);
      }
      List<HColumn<Long,Float>> itemIDColumns = result.getColumns();
      if (itemIDColumns.isEmpty()) {
        throw new NoSuchUserException(userID);
      }
      int size = itemIDColumns.size();
      PreferenceArray prefs = new GenericUserPreferenceArray(size);
      prefs.setUserID(0, userID);
      for (int i = 0; i < size; i++) {
        HColumn<Long,Float> itemIDColumn = itemIDColumns.get(i);
        prefs.setItemID(i, itemIDColumn.getName());
        prefs.setValue(i, itemIDColumn.getValue());
      }
      return prefs;
    }
  }

  private final class ItemPrefArrayRetriever implements Retriever<Long, PreferenceArray> {
    @Override
    public PreferenceArray get(Long itemID) throws TasteException {
      SliceQuery<Long,Long,Float> query = buildValueSliceQuery(ITEMS_CF);
      query.setKey(itemID);
      ColumnSlice<Long,Float> result = query.execute().get();
      if (result == null) {
        throw new NoSuchItemException(itemID);
      }
      List<HColumn<Long,Float>> userIDColumns = result.getColumns();
      if (userIDColumns.isEmpty()) {
        throw new NoSuchItemException(itemID);
      }
      int size = userIDColumns.size();
      PreferenceArray prefs = new GenericItemPreferenceArray(size);
      prefs.setItemID(0, itemID);
      for (int i = 0; i < size; i++) {
        HColumn<Long,Float> userIDColumn = userIDColumns.get(i);
        prefs.setUserID(i, userIDColumn.getName());
        prefs.setValue(i, userIDColumn.getValue());
      }
      return prefs;
    }
  }

  private final class UserIDsFromItemRetriever implements Retriever<Long, FastIDSet> {
    @Override
    public FastIDSet get(Long itemID) throws TasteException {
      SliceQuery<Long,Long,byte[]> query = buildNoValueSliceQuery(ITEMS_CF);
      query.setKey(itemID);
      ColumnSlice<Long,byte[]> result = query.execute().get();
      if (result == null) {
        throw new NoSuchItemException(itemID);
      }
      List<HColumn<Long,byte[]>> columns = result.getColumns();
      FastIDSet userIDs = new FastIDSet(columns.size());
      for (HColumn<Long,?> userIDColumn : columns) {
        userIDs.add(userIDColumn.getName());
      }
      return userIDs;
    }
  }

  private final class ItemIDsFromUserRetriever implements Retriever<Long, FastIDSet> {
    @Override
    public FastIDSet get(Long userID) throws TasteException {
      SliceQuery<Long,Long,byte[]> query = buildNoValueSliceQuery(USERS_CF);
      query.setKey(userID);
      FastIDSet itemIDs = new FastIDSet();
      ColumnSlice<Long,byte[]> result = query.execute().get();
      if (result == null) {
        throw new NoSuchUserException(userID);
      }
      List<HColumn<Long,byte[]>> columns = result.getColumns();
      if (columns.isEmpty()) {
        throw new NoSuchUserException(userID);
      }
      for (HColumn<Long,?> itemIDColumn : columns) {
        itemIDs.add(itemIDColumn.getName());
      }
      return itemIDs;
    }
  }

}
