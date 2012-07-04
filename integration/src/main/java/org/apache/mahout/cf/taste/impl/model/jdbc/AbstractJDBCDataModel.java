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

package org.apache.mahout.cf.taste.impl.model.jdbc;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Collection;
import java.util.List;

import javax.sql.DataSource;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.cf.taste.impl.common.jdbc.ResultSetIterator;
import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * An abstract superclass for {@link JDBCDataModel} implementations, providing most of the common
 * functionality that any such implementation would need.
 * </p>
 *
 * <p>
 * Performance will be a concern with any {@link JDBCDataModel}. There are going to be lots of
 * simultaneous reads and some writes to one table. Make sure the table is set up optimally -- for example,
 * you'll want to establish indexes.
 * </p>
 *
 * <p>
 * You'll also want to use connection pooling of some kind. Most J2EE containers like Tomcat provide
 * connection pooling, so make sure the {@link DataSource} it exposes is using pooling. Outside a J2EE
 * container, you can use packages like Jakarta's <a href="http://jakarta.apache.org/commons/dbcp/">DBCP</a>
 * to create a {@link DataSource} on top of your database whose {@link Connection}s are pooled.
 * </p>
 */
public abstract class AbstractJDBCDataModel extends AbstractJDBCComponent implements JDBCDataModel {

  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCDataModel.class);

  public static final String DEFAULT_PREFERENCE_TABLE = "taste_preferences";
  public static final String DEFAULT_USER_ID_COLUMN = "user_id";
  public static final String DEFAULT_ITEM_ID_COLUMN = "item_id";
  public static final String DEFAULT_PREFERENCE_COLUMN = "preference";
  public static final String DEFAULT_PREFERENCE_TIME_COLUMN = "timestamp";

  private final DataSource dataSource;
  private final String preferenceTable;
  private final String userIDColumn;
  private final String itemIDColumn;
  private final String preferenceColumn;
  private final String getPreferenceSQL;
  private final String getPreferenceTimeSQL;
  private final String getUserSQL;
  private final String getAllUsersSQL;
  private final String getNumItemsSQL;
  private final String getNumUsersSQL;
  private final String setPreferenceSQL;
  private final String removePreferenceSQL;
  private final String getUsersSQL;
  private final String getItemsSQL;
  private final String getPrefsForItemSQL;
  //private final String getNumPreferenceForItemSQL;
  private final String getNumPreferenceForItemsSQL;
  private final String getMaxPreferenceSQL;
  private final String getMinPreferenceSQL;
  private int cachedNumUsers;
  private int cachedNumItems;
  private final Cache<Long,Integer> itemPrefCounts;
  private float maxPreference;
  private float minPreference;

  protected AbstractJDBCDataModel(DataSource dataSource,
                                  String getPreferenceSQL,
                                  String getPreferenceTimeSQL,
                                  String getUserSQL,
                                  String getAllUsersSQL,
                                  String getNumItemsSQL,
                                  String getNumUsersSQL,
                                  String setPreferenceSQL,
                                  String removePreferenceSQL,
                                  String getUsersSQL,
                                  String getItemsSQL,
                                  String getPrefsForItemSQL,
                                  String getNumPreferenceForItemSQL,
                                  String getNumPreferenceForItemsSQL,
                                  String getMaxPreferenceSQL,
                                  String getMinPreferenceSQL) {
    this(dataSource,
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_COLUMN,
         getPreferenceSQL,
         getPreferenceTimeSQL,
         getUserSQL,
         getAllUsersSQL,
         getNumItemsSQL,
         getNumUsersSQL,
         setPreferenceSQL,
         removePreferenceSQL,
         getUsersSQL,
         getItemsSQL,
         getPrefsForItemSQL,
         getNumPreferenceForItemSQL,
         getNumPreferenceForItemsSQL,
         getMaxPreferenceSQL,
         getMinPreferenceSQL);
  }

  protected AbstractJDBCDataModel(DataSource dataSource,
                                  String preferenceTable,
                                  String userIDColumn,
                                  String itemIDColumn,
                                  String preferenceColumn,
                                  String getPreferenceSQL,
                                  String getPreferenceTimeSQL,
                                  String getUserSQL,
                                  String getAllUsersSQL,
                                  String getNumItemsSQL,
                                  String getNumUsersSQL,
                                  String setPreferenceSQL,
                                  String removePreferenceSQL,
                                  String getUsersSQL,
                                  String getItemsSQL,
                                  String getPrefsForItemSQL,
                                  String getNumPreferenceForItemSQL,
                                  String getNumPreferenceForItemsSQL,
                                  String getMaxPreferenceSQL,
                                  String getMinPreferenceSQL) {

    log.debug("Creating AbstractJDBCModel...");

    AbstractJDBCComponent.checkNotNullAndLog("preferenceTable", preferenceTable);
    AbstractJDBCComponent.checkNotNullAndLog("userIDColumn", userIDColumn);
    AbstractJDBCComponent.checkNotNullAndLog("itemIDColumn", itemIDColumn);
    AbstractJDBCComponent.checkNotNullAndLog("preferenceColumn", preferenceColumn);

    AbstractJDBCComponent.checkNotNullAndLog("dataSource", dataSource);
    AbstractJDBCComponent.checkNotNullAndLog("getUserSQL", getUserSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getAllUsersSQL", getAllUsersSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getPreferenceSQL", getPreferenceSQL);
    // getPreferenceTimeSQL can be null
    AbstractJDBCComponent.checkNotNullAndLog("getNumItemsSQL", getNumItemsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getNumUsersSQL", getNumUsersSQL);
    AbstractJDBCComponent.checkNotNullAndLog("setPreferenceSQL", setPreferenceSQL);
    AbstractJDBCComponent.checkNotNullAndLog("removePreferenceSQL", removePreferenceSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getUsersSQL", getUsersSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getItemsSQL", getItemsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getPrefsForItemSQL", getPrefsForItemSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getNumPreferenceForItemSQL", getNumPreferenceForItemSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getNumPreferenceForItemsSQL", getNumPreferenceForItemsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getMaxPreferenceSQL", getMaxPreferenceSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getMinPreferenceSQL", getMinPreferenceSQL);

    if (!(dataSource instanceof ConnectionPoolDataSource)) {
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections "
                + "to the database itself, or database performance will be severely reduced.");
    }

    this.preferenceTable = preferenceTable;
    this.userIDColumn = userIDColumn;
    this.itemIDColumn = itemIDColumn;
    this.preferenceColumn = preferenceColumn;

    this.dataSource = dataSource;
    this.getPreferenceSQL = getPreferenceSQL;
    this.getPreferenceTimeSQL = getPreferenceTimeSQL;
    this.getUserSQL = getUserSQL;
    this.getAllUsersSQL = getAllUsersSQL;
    this.getNumItemsSQL = getNumItemsSQL;
    this.getNumUsersSQL = getNumUsersSQL;
    this.setPreferenceSQL = setPreferenceSQL;
    this.removePreferenceSQL = removePreferenceSQL;
    this.getUsersSQL = getUsersSQL;
    this.getItemsSQL = getItemsSQL;
    this.getPrefsForItemSQL = getPrefsForItemSQL;
    //this.getNumPreferenceForItemSQL = getNumPreferenceForItemSQL;
    this.getNumPreferenceForItemsSQL = getNumPreferenceForItemsSQL;
    this.getMaxPreferenceSQL = getMaxPreferenceSQL;
    this.getMinPreferenceSQL = getMinPreferenceSQL;

    this.cachedNumUsers = -1;
    this.cachedNumItems = -1;
    this.itemPrefCounts = new Cache<Long,Integer>(new ItemPrefCountRetriever(getNumPreferenceForItemSQL));

    this.maxPreference = Float.NaN;
    this.minPreference = Float.NaN;
  }

  /** @return the {@link DataSource} that this instance is using */
  @Override
  public DataSource getDataSource() {
    return dataSource;
  }

  public String getPreferenceTable() {
    return preferenceTable;
  }

  public String getUserIDColumn() {
    return userIDColumn;
  }

  public String getItemIDColumn() {
    return itemIDColumn;
  }

  public String getPreferenceColumn() {
    return preferenceColumn;
  }

  String getSetPreferenceSQL() {
    return setPreferenceSQL;
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    log.debug("Retrieving all users...");
    try {
      return new ResultSetIDIterator(getUsersSQL);
    } catch (SQLException sqle) {
      throw new TasteException(sqle);
    }
  }

  /**
   * @throws NoSuchUserException
   *           if there is no such user
   */
  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {

    log.debug("Retrieving user ID '{}'", userID);

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getUserSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      setLongParameter(stmt, 1, userID);

      log.debug("Executing SQL query: {}", getUserSQL);
      rs = stmt.executeQuery();

      List<Preference> prefs = Lists.newArrayList();
      while (rs.next()) {
        prefs.add(buildPreference(rs));
      }

      if (prefs.isEmpty()) {
        throw new NoSuchUserException(userID);
      }

      return new GenericUserPreferenceArray(prefs);

    } catch (SQLException sqle) {
      log.warn("Exception while retrieving user", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }

  }

  @Override
  public FastByIDMap<PreferenceArray> exportWithPrefs() throws TasteException {
    log.debug("Exporting all data");

    Connection conn = null;
    Statement stmt = null;
    ResultSet rs = null;

    FastByIDMap<PreferenceArray> result = new FastByIDMap<PreferenceArray>();

    try {
      conn = dataSource.getConnection();
      stmt = conn.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());

      log.debug("Executing SQL query: {}", getAllUsersSQL);
      rs = stmt.executeQuery(getAllUsersSQL);

      Long currentUserID = null;
      List<Preference> currentPrefs = Lists.newArrayList();
      while (rs.next()) {
        long nextUserID = getLongColumn(rs, 1);
        if (currentUserID != null && !currentUserID.equals(nextUserID) && !currentPrefs.isEmpty()) {
          result.put(currentUserID, new GenericUserPreferenceArray(currentPrefs));
          currentPrefs.clear();
        }
        currentPrefs.add(buildPreference(rs));
        currentUserID = nextUserID;
      }
      if (!currentPrefs.isEmpty()) {
        result.put(currentUserID, new GenericUserPreferenceArray(currentPrefs));
      }

      return result;

    } catch (SQLException sqle) {
      log.warn("Exception while exporting all data", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);

    }
  }

  @Override
  public FastByIDMap<FastIDSet> exportWithIDsOnly() throws TasteException {
    log.debug("Exporting all data");

    Connection conn = null;
    Statement stmt = null;
    ResultSet rs = null;

    FastByIDMap<FastIDSet> result = new FastByIDMap<FastIDSet>();

    try {
      conn = dataSource.getConnection();
      stmt = conn.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());

      log.debug("Executing SQL query: {}", getAllUsersSQL);
      rs = stmt.executeQuery(getAllUsersSQL);

      boolean currentUserIDSet = false;
      long currentUserID = 0L; // value isn't used
      FastIDSet currentItemIDs = new FastIDSet(2);
      while (rs.next()) {
        long nextUserID = getLongColumn(rs, 1);
        if (currentUserIDSet && currentUserID != nextUserID && !currentItemIDs.isEmpty()) {
          result.put(currentUserID, currentItemIDs);
          currentItemIDs = new FastIDSet(2);
        }
        currentItemIDs.add(getLongColumn(rs, 2));
        currentUserID = nextUserID;
        currentUserIDSet = true;
      }
      if (!currentItemIDs.isEmpty()) {
        result.put(currentUserID, currentItemIDs);
      }

      return result;

    } catch (SQLException sqle) {
      log.warn("Exception while exporting all data", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);

    }
  }

  /**
   * @throws NoSuchUserException
   *           if there is no such user
   */
  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {

    log.debug("Retrieving items for user ID '{}'", userID);

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getUserSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      setLongParameter(stmt, 1, userID);

      log.debug("Executing SQL query: {}", getUserSQL);
      rs = stmt.executeQuery();

      FastIDSet result = new FastIDSet();
      while (rs.next()) {
        result.add(getLongColumn(rs, 2));
      }

      if (result.isEmpty()) {
        throw new NoSuchUserException(userID);
      }

      return result;

    } catch (SQLException sqle) {
      log.warn("Exception while retrieving item s", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }

  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    log.debug("Retrieving preferences for item ID '{}'", itemID);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getPreferenceSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(1);
      setLongParameter(stmt, 1, userID);
      setLongParameter(stmt, 2, itemID);

      log.debug("Executing SQL query: {}", getPreferenceSQL);
      rs = stmt.executeQuery();
      if (rs.next()) {
        return rs.getFloat(1);
      } else {
        return null;
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving prefs for item", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    if (getPreferenceTimeSQL == null) {
      return null;
    }
    log.debug("Retrieving preference time for item ID '{}'", itemID);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getPreferenceTimeSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(1);
      setLongParameter(stmt, 1, userID);
      setLongParameter(stmt, 2, itemID);

      log.debug("Executing SQL query: {}", getPreferenceTimeSQL);
      rs = stmt.executeQuery();
      if (rs.next()) {
        return rs.getLong(1);
      } else {
        return null;
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving time for item", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    log.debug("Retrieving all items...");
    try {
      return new ResultSetIDIterator(getItemsSQL);
    } catch (SQLException sqle) {
      throw new TasteException(sqle);
    }
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    List<Preference> list = doGetPreferencesForItem(itemID);
    if (list.isEmpty()) {
      throw new NoSuchItemException(itemID);
    }
    return new GenericItemPreferenceArray(list);
  }

  protected List<Preference> doGetPreferencesForItem(long itemID) throws TasteException {
    log.debug("Retrieving preferences for item ID '{}'", itemID);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getPrefsForItemSQL, ResultSet.TYPE_FORWARD_ONLY,
        ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      setLongParameter(stmt, 1, itemID);

      log.debug("Executing SQL query: {}", getPrefsForItemSQL);
      rs = stmt.executeQuery();
      List<Preference> prefs = Lists.newArrayList();
      while (rs.next()) {
        prefs.add(buildPreference(rs));
      }
      return prefs;
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving prefs for item", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public int getNumItems() throws TasteException {
    if (cachedNumItems < 0) {
      cachedNumItems = getNumThings("items", getNumItemsSQL);
    }
    return cachedNumItems;
  }

  @Override
  public int getNumUsers() throws TasteException {
    if (cachedNumUsers < 0) {
      cachedNumUsers = getNumThings("users", getNumUsersSQL);
    }
    return cachedNumUsers;
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return itemPrefCounts.get(itemID);
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    return getNumThings("user preferring items", getNumPreferenceForItemsSQL, itemID1, itemID2);
  }

  private int getNumThings(String name, String sql, long... args) throws TasteException {
    log.debug("Retrieving number of {} in model", name);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(sql, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      if (args != null) {
        for (int i = 1; i <= args.length; i++) {
          setLongParameter(stmt, i, args[i - 1]);
        }
      }
      log.debug("Executing SQL query: {}", sql);
      rs = stmt.executeQuery();
      rs.next();
      return rs.getInt(1);
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving number of {}", name, sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    Preconditions.checkArgument(!Float.isNaN(value), "NaN value");

    log.debug("Setting preference for user {}, item {}", userID, itemID);

    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(setPreferenceSQL);
      setLongParameter(stmt, 1, userID);
      setLongParameter(stmt, 2, itemID);
      stmt.setDouble(3, value);
      stmt.setDouble(4, value);

      log.debug("Executing SQL update: {}", setPreferenceSQL);
      stmt.executeUpdate();

    } catch (SQLException sqle) {
      log.warn("Exception while setting preference", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {

    log.debug("Removing preference for user '{}', item '{}'", userID, itemID);

    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(removePreferenceSQL);
      setLongParameter(stmt, 1, userID);
      setLongParameter(stmt, 2, itemID);

      log.debug("Executing SQL update: {}", removePreferenceSQL);
      stmt.executeUpdate();

    } catch (SQLException sqle) {
      log.warn("Exception while removing preference", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    cachedNumUsers = -1;
    cachedNumItems = -1;
    minPreference = Float.NaN;
    maxPreference = Float.NaN;
    itemPrefCounts.clear();
  }

  @Override
  public boolean hasPreferenceValues() {
    return true;
  }

  @Override
  public float getMaxPreference() {
    if (Float.isNaN(maxPreference)) {
      Connection conn = null;
      PreparedStatement stmt = null;
      ResultSet rs = null;
      try {
        conn = dataSource.getConnection();
        stmt = conn.prepareStatement(getMaxPreferenceSQL);

        log.debug("Executing SQL query: {}", getMaxPreferenceSQL);
        rs = stmt.executeQuery();
        rs.next();
        maxPreference = rs.getFloat(1);

      } catch (SQLException sqle) {
        log.warn("Exception while removing preference", sqle);
        // do nothing
      } finally {
        IOUtils.quietClose(rs, stmt, conn);
      }
    }
    return maxPreference;
  }

  @Override
  public float getMinPreference() {
    if (Float.isNaN(minPreference)) {
      Connection conn = null;
      PreparedStatement stmt = null;
      ResultSet rs = null;
      try {
        conn = dataSource.getConnection();
        stmt = conn.prepareStatement(getMinPreferenceSQL);

        log.debug("Executing SQL query: {}", getMinPreferenceSQL);
        rs = stmt.executeQuery();
        rs.next();
        minPreference = rs.getFloat(1);

      } catch (SQLException sqle) {
        log.warn("Exception while removing preference", sqle);
        // do nothing
      } finally {
        IOUtils.quietClose(rs, stmt, conn);
      }
    }
    return minPreference;
  }

  // Some overrideable methods to customize the class behavior:

  protected Preference buildPreference(ResultSet rs) throws SQLException {
    return new GenericPreference(getLongColumn(rs, 1), getLongColumn(rs, 2), rs.getFloat(3));
  }

  /**
   * Subclasses may wish to override this if ID values in the file are not numeric. This provides a hook by
   * which subclasses can inject an {@link org.apache.mahout.cf.taste.model.IDMigrator} to perform
   * translation.
   */
  protected long getLongColumn(ResultSet rs, int position) throws SQLException {
    return rs.getLong(position);
  }

  /**
   * Subclasses may wish to override this if ID values in the file are not numeric. This provides a hook by
   * which subclasses can inject an {@link org.apache.mahout.cf.taste.model.IDMigrator} to perform
   * translation.
   */
  protected void setLongParameter(PreparedStatement stmt, int position, long value) throws SQLException {
    stmt.setLong(position, value);
  }

  /**
   * <p>
   * An {@link java.util.Iterator} which returns items from a {@link ResultSet}. This is a useful way to
   * iterate over all user data since it does not require all data to be read into memory at once. It does
   * however require that the DB connection be held open. Note that this class will only release database
   * resources after {@link #hasNext()} has been called and has returned {@code false}; callers should
   * make sure to "drain" the entire set of data to avoid tying up database resources.
   * </p>
   */
  private final class ResultSetIDIterator extends ResultSetIterator<Long> implements LongPrimitiveIterator {

    private ResultSetIDIterator(String sql) throws SQLException {
      super(dataSource, sql);
    }

    @Override
    protected Long parseElement(ResultSet resultSet) throws SQLException {
      return getLongColumn(resultSet, 1);
    }

    @Override
    public long nextLong() {
      return next();
    }

    /**
     * @throws UnsupportedOperationException
     */
    @Override
    public long peek() {
      // This could be supported; is it worth it?
      throw new UnsupportedOperationException();
    }
  }

  private final class ItemPrefCountRetriever implements Retriever<Long,Integer> {
    private final String getNumPreferenceForItemSQL;

    private ItemPrefCountRetriever(String getNumPreferenceForItemSQL) {
      this.getNumPreferenceForItemSQL = getNumPreferenceForItemSQL;
    }

    @Override
    public Integer get(Long key) throws TasteException {
      return getNumThings("user preferring item", getNumPreferenceForItemSQL, key);
    }
  }
}
