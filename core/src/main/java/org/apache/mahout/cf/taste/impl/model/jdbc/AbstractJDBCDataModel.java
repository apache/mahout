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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.naming.Context;
import javax.naming.InitialContext;
import javax.naming.NamingException;
import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * <p>An abstract superclass for JDBC-related {@link DataModel} implementations, providing most of the common
 * functionality that any such implementation would need.</p>
 *
 * <p>Performance will be a concern with any JDBC-based {@link DataModel}. There are going to be lots of
 * simultaneous reads and some writes to one table. Make sure the table is set up optimally -- for example,
 * you'll want to establish indexes.</p>
 *
 * <p>You'll also want to use connection pooling of some kind. Most J2EE containers like Tomcat
 * provide connection pooling, so make sure the {@link DataSource} it exposes is using pooling. Outside a
 * J2EE container, you can use packages like Jakarta's
 * <a href="http://jakarta.apache.org/commons/dbcp/">DBCP</a> to create a {@link DataSource} on top of your
 * database whose {@link Connection}s are pooled.</p>
 *
 * <p>Also note: this default implementation assumes that the user and item ID keys are {@link String}s, for
 * maximum flexibility. You can override this behavior by subclassing an implementation and overriding
 * {@link #buildItem(String)} and {@link #buildUser(String, List)}. If you don't, just make sure you use
 * {@link String}s as IDs throughout your code. If your IDs are really numeric, and you use, say, {@link Long}
 * for IDs in the rest of your code, you will run into subtle problems because the {@link Long} values won't
 * be equal to or compare correctly to the underlying {@link String} key values.</p>
 */
public abstract class AbstractJDBCDataModel implements JDBCDataModel {

  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCDataModel.class);

  public static final String DEFAULT_DATASOURCE_NAME = "jdbc/taste";
  public static final String DEFAULT_PREFERENCE_TABLE = "taste_preferences";
  public static final String DEFAULT_USER_ID_COLUMN = "user_id";
  public static final String DEFAULT_ITEM_ID_COLUMN = "item_id";
  public static final String DEFAULT_PREFERENCE_COLUMN = "preference";

  private final DataSource dataSource;
  private final String getUserSQL;
  private final String getNumItemsSQL;
  private final String getNumUsersSQL;
  private final String setPreferenceSQL;
  private final String removePreferenceSQL;
  private final String getUsersSQL;
  private final String getItemsSQL;
  private final String getItemSQL;
  private final String getPrefsForItemSQL;
  private final String getNumPreferenceForItemSQL;
  private final String getNumPreferenceForItemsSQL;

  protected AbstractJDBCDataModel(DataSource dataSource,
                                  String getUserSQL,
                                  String getNumItemsSQL,
                                  String getNumUsersSQL,
                                  String setPreferenceSQL,
                                  String removePreferenceSQL,
                                  String getUsersSQL,
                                  String getItemsSQL,
                                  String getItemSQL,
                                  String getPrefsForItemSQL,
                                  String getNumPreferenceForItemSQL,
                                  String getNumPreferenceForItemsSQL) {

    log.debug("Creating AbstractJDBCModel...");
    checkNotNullAndLog("dataSource", dataSource);
    checkNotNullAndLog("getUserSQL", getUserSQL);
    checkNotNullAndLog("getNumItemsSQL", getNumItemsSQL);
    checkNotNullAndLog("getNumUsersSQL", getNumUsersSQL);
    checkNotNullAndLog("setPreferenceSQL", setPreferenceSQL);
    checkNotNullAndLog("removePreferenceSQL", removePreferenceSQL);
    checkNotNullAndLog("getUsersSQL", getUsersSQL);
    checkNotNullAndLog("getItemsSQL", getItemsSQL);
    checkNotNullAndLog("getItemSQL", getItemSQL);
    checkNotNullAndLog("getPrefsForItemSQL", getPrefsForItemSQL);
    checkNotNullAndLog("getNumPreferenceForItemSQL", getNumPreferenceForItemSQL);
    checkNotNullAndLog("getNumPreferenceForItemsSQL", getNumPreferenceForItemsSQL);

    if (!(dataSource instanceof ConnectionPoolDataSource)) {
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections " +
               "to the database itself, or database performance will be severely reduced.");
    }

    this.dataSource = dataSource;
    this.getUserSQL = getUserSQL;
    this.getNumItemsSQL = getNumItemsSQL;
    this.getNumUsersSQL = getNumUsersSQL;
    this.setPreferenceSQL = setPreferenceSQL;
    this.removePreferenceSQL = removePreferenceSQL;
    this.getUsersSQL = getUsersSQL;
    this.getItemsSQL = getItemsSQL;
    this.getItemSQL = getItemSQL;
    this.getPrefsForItemSQL = getPrefsForItemSQL;
    this.getNumPreferenceForItemSQL = getNumPreferenceForItemSQL;
    this.getNumPreferenceForItemsSQL = getNumPreferenceForItemsSQL;
  }

  private static void checkNotNullAndLog(String argName, Object value) {
    if (value == null || value.toString().length() == 0) {
      throw new IllegalArgumentException(argName + " is null or empty");
    }
    log.debug("{}: {}", argName, value);
  }

  /**
   * <p>Looks up a {@link DataSource} by name from JNDI. "java:comp/env/" is prepended to the argument
   * before looking up the name in JNDI.</p>
   *
   * @param dataSourceName JNDI name where a {@link DataSource} is bound (e.g. "jdbc/taste")
   * @return {@link DataSource} under that JNDI name
   * @throws TasteException if a JNDI error occurs
   */
  public static DataSource lookupDataSource(String dataSourceName) throws TasteException {
    Context context = null;
    try {
      context = new InitialContext();
      return (DataSource) context.lookup("java:comp/env/" + dataSourceName);
    } catch (NamingException ne) {
      throw new TasteException(ne);
    } finally {
      if (context != null) {
        try {
          context.close();
        } catch (NamingException ne) {
          log.warn("Error while closing Context; continuing...", ne);
        }
      }
    }
  }

  /**
   * @return the {@link DataSource} that this instance is using
   */
  public DataSource getDataSource() {
    return dataSource;
  }

  public final Iterable<? extends User> getUsers() throws TasteException {
    log.debug("Retrieving all users...");
    return new IteratorIterable<User>(new ResultSetUserIterator(dataSource, getUsersSQL));
  }

  /**
   * @throws NoSuchElementException if there is no such user
   */
  public final User getUser(Object id) throws TasteException {

    log.debug("Retrieving user ID '{}'", id);

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    String idString = id.toString();

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getUserSQL);
      stmt.setObject(1, id);

      log.debug("Executing SQL query: {}", getUserSQL);
      rs = stmt.executeQuery();

      List<Preference> prefs = new ArrayList<Preference>();
      while (rs.next()) {
        addPreference(rs, prefs);
      }

      if (prefs.isEmpty()) {
        throw new NoSuchElementException();
      }

      return buildUser(idString, prefs);

    } catch (SQLException sqle) {
      log.warn("Exception while retrieving user", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }

  }

  public final Iterable<? extends Item> getItems() throws TasteException {
    log.debug("Retrieving all items...");
    return new IteratorIterable<Item>(new ResultSetItemIterator(dataSource, getItemsSQL));
  }

  public final Item getItem(Object id) throws TasteException {
    return getItem(id, false);
  }

  public final Item getItem(Object id, boolean assumeExists) throws TasteException {

    if (assumeExists) {
      return buildItem((String) id);
    }

    log.debug("Retrieving item ID '{}'", id);

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getItemSQL);
      stmt.setObject(1, id);

      log.debug("Executing SQL query: {}", getItemSQL);
      rs = stmt.executeQuery();
      if (rs.next()) {
        return buildItem((String) id);
      } else {
        throw new NoSuchElementException();
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving item", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  public final Iterable<? extends Preference> getPreferencesForItem(Object itemID) throws TasteException {
    return doGetPreferencesForItem(itemID);
  }

  public final Preference[] getPreferencesForItemAsArray(Object itemID) throws TasteException {
    List<? extends Preference> list = doGetPreferencesForItem(itemID);
    return list.toArray(new Preference[list.size()]);
  }

  private List<? extends Preference> doGetPreferencesForItem(Object itemID) throws TasteException {
    log.debug("Retrieving preferences for item ID '{}'", itemID);
    Item item = getItem(itemID, true);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getPrefsForItemSQL);
      stmt.setObject(1, itemID);

      log.debug("Executing SQL query: {}", getPrefsForItemSQL);
      rs = stmt.executeQuery();
      List<Preference> prefs = new ArrayList<Preference>();
      while (rs.next()) {
        double preference = rs.getDouble(1);
        String userID = rs.getString(2);
        Preference pref = buildPreference(buildUser(userID, null), item, preference);
        prefs.add(pref);
      }
      return prefs;
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving prefs for item", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  public final int getNumItems() throws TasteException {
    return getNumThings("items", getNumItemsSQL);
  }

  public final int getNumUsers() throws TasteException {
    return getNumThings("users", getNumUsersSQL);
  }

  public final int getNumUsersWithPreferenceFor(Object... itemIDs) throws TasteException {
    if (itemIDs == null) {
      throw new IllegalArgumentException("itemIDs is null");
    }
    int length = itemIDs.length;
    if (length == 0 || length > 2) {
      throw new IllegalArgumentException("Illegal number of item IDs: " + length);
    }
    if (length == 1) {
      return getNumThings("user preferring item", getNumPreferenceForItemSQL, itemIDs);
    } else {
      return getNumThings("user preferring items", getNumPreferenceForItemsSQL, itemIDs);
    }
  }


  private int getNumThings(String name, String sql, Object... args) throws TasteException {
    log.debug("Retrieving number of {} in model", name);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(sql);
      if (args != null) {
        for (int i = 1; i <= args.length; i++) {
          stmt.setObject(i, args[i - 1]);
        }
      }
      log.debug("Executing SQL query: {}", sql);
      rs = stmt.executeQuery();
      rs.next();
      return rs.getInt(1);
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving number of " + name, sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  public final void setPreference(Object userID, Object itemID, double value)
          throws TasteException {
    if (userID == null || itemID == null) {
      throw new IllegalArgumentException("userID or itemID is null");
    }
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }

    if (log.isDebugEnabled()) {
      log.debug("Setting preference for user '" + userID + "', item '" + itemID + "', value " + value);
    }

    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = dataSource.getConnection();

      stmt = conn.prepareStatement(setPreferenceSQL);
      stmt.setObject(1, userID);
      stmt.setObject(2, itemID);
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

  public final void removePreference(Object userID, Object itemID)
          throws TasteException {
    if (userID == null || itemID == null) {
      throw new IllegalArgumentException("userID or itemID is null");
    }

    log.debug("Removing preference for user '{}', item '{}'", userID, itemID);

    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = dataSource.getConnection();

      stmt = conn.prepareStatement(removePreferenceSQL);
      stmt.setObject(1, userID);
      stmt.setObject(2, itemID);

      log.debug("Executing SQL update: {}", removePreferenceSQL);
      stmt.executeUpdate();

    } catch (SQLException sqle) {
      log.warn("Exception while removing preference", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }

  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }


  private void addPreference(ResultSet rs, Collection<Preference> prefs)
          throws SQLException {
    Item item = buildItem(rs.getString(1));
    double preferenceValue = rs.getDouble(2);
    prefs.add(buildPreference(null, item, preferenceValue));
  }

  /**
   * <p>Default implementation which returns a new {@link GenericUser} with {@link String} IDs.
   * Subclasses may override to return a different {@link User} implementation.</p>
   *
   * @param id user ID
   * @param prefs user preferences
   * @return {@link GenericUser} by default
   */
  protected User buildUser(String id, List<Preference> prefs) {
    return new GenericUser<String>(id, prefs);
  }

  /**
   * <p>Default implementation which returns a new {@link GenericItem} with {@link String} IDs.
   * Subclasses may override to return a different {@link Item} implementation.</p>
   *
   * @param id item ID
   * @return {@link GenericItem} by default
   */
  protected Item buildItem(String id) {
    return new GenericItem<String>(id);
  }

  /**
   * Subclasses may override to return a different {@link Preference} implementation.
   *
   * @param user {@link User}
   * @param item {@link Item}
   * @return {@link GenericPreference} by default
   */
  protected Preference buildPreference(User user, Item item, double value) {
    return new GenericPreference(user, item, value);
  }

  /**
   * <p>An {@link java.util.Iterator} which returns {@link org.apache.mahout.cf.taste.model.User}s from a
   * {@link java.sql.ResultSet}. This is a useful
   * way to iterate over all user data since it does not require all data to be read into memory
   * at once. It does however require that the DB connection be held open. Note that this class will
   * only release database resources after {@link #hasNext()} has been called and has returned false;
   * callers should make sure to "drain" the entire set of data to avoid tying up database resources.</p>
   */
  private final class ResultSetUserIterator implements Iterator<User> {

    private final Connection connection;
    private final PreparedStatement statement;
    private final ResultSet resultSet;
    private boolean closed;

    private ResultSetUserIterator(DataSource dataSource, String getUsersSQL) throws TasteException {
      try {
        connection = dataSource.getConnection();
        statement = connection.prepareStatement(getUsersSQL);
        statement.setFetchDirection(ResultSet.FETCH_UNKNOWN);
        log.debug("Executing SQL query: {}", getUsersSQL);
        resultSet = statement.executeQuery();
      } catch (SQLException sqle) {
        close();
        throw new TasteException(sqle);
      }
    }

    public boolean hasNext() {
      boolean nextExists = false;
      if (!closed) {
        try {
          // No more results if cursor is pointing at last row, or after
          // Thanks to Rolf W. for pointing out an earlier bug in this condition
          if (resultSet.isLast() || resultSet.isAfterLast()) {
            close();
          } else {
            nextExists = true;
          }
        } catch (SQLException sqle) {
          log.warn("Unexpected exception while accessing ResultSet; continuing...", sqle);
          close();
        }
      }
      return nextExists;
    }

    public User next() {

      if (closed) {
        throw new NoSuchElementException();
      }

      String currentUserID = null;
      List<Preference> prefs = new ArrayList<Preference>();

      try {
        while (resultSet.next()) {
          String userID = resultSet.getString(3);
          if (currentUserID == null) {
            currentUserID = userID;
          }
          // Did we move on to a new user?
          if (!userID.equals(currentUserID)) {
            // back up one row
            resultSet.previous();
            // we're done for now
            break;
          }
          // else add a new preference for the current user
          addPreference(resultSet, prefs);
        }
      } catch (SQLException sqle) {
        // No good way to handle this since we can't throw an exception
        log.warn("Exception while iterating over users", sqle);
        close();
        throw new NoSuchElementException("Can't retrieve more due to exception: " + sqle);
      }

      if (currentUserID == null) {
        // nothing left?
        throw new NoSuchElementException();
      }

      return buildUser(currentUserID, prefs);
    }

    /**
     * @throws UnsupportedOperationException
     */
    public void remove() {
      throw new UnsupportedOperationException();
    }

    private void close() {
      closed = true;
      IOUtils.quietClose(resultSet, statement, connection);
    }

  }

  /**
   * <p>An {@link java.util.Iterator} which returns {@link org.apache.mahout.cf.taste.model.Item}s from a
   * {@link java.sql.ResultSet}. This is a useful way to iterate over all user data since it does not require
   * all data to be read into memory at once. It does however require that the DB connection be held open. Note
   * that this class will only release database resources after {@link #hasNext()} has been called and has returned
   * <code>false</code>; callers should make sure to "drain" the entire set of data to avoid tying up database
   * resources.</p>
   */
  private final class ResultSetItemIterator implements Iterator<Item> {

    private final Connection connection;
    private final PreparedStatement statement;
    private final ResultSet resultSet;
    private boolean closed;

    private ResultSetItemIterator(DataSource dataSource, String getItemsSQL) throws TasteException {
      try {
        connection = dataSource.getConnection();
        statement = connection.prepareStatement(getItemsSQL);
        statement.setFetchDirection(ResultSet.FETCH_FORWARD);
        log.debug("Executing SQL query: {}", getItemsSQL);
        resultSet = statement.executeQuery();
      } catch (SQLException sqle) {
        close();
        throw new TasteException(sqle);
      }
    }

    public boolean hasNext() {
      boolean nextExists = false;
      if (!closed) {
        try {
          // No more results if cursor is pointing at last row, or after
          // Thanks to Rolf W. for pointing out an earlier bug in this condition
          if (resultSet.isLast() || resultSet.isAfterLast()) {
            close();
          } else {
            nextExists = true;
          }
        } catch (SQLException sqle) {
          log.warn("Unexpected exception while accessing ResultSet; continuing...", sqle);
          close();
        }
      }
      return nextExists;
    }

    public Item next() {

      if (closed) {
        throw new NoSuchElementException();
      }

      try {
        if (resultSet.next()) {
          return buildItem(resultSet.getString(1));
        } else {
          throw new NoSuchElementException();
        }
      } catch (SQLException sqle) {
        // No good way to handle this since we can't throw an exception
        log.warn("Exception while iterating over items", sqle);
        close();
        throw new NoSuchElementException("Can't retrieve more due to exception: " + sqle);
      }

    }

    /**
     * @throws UnsupportedOperationException
     */
    public void remove() {
      throw new UnsupportedOperationException();
    }

    private void close() {
      closed = true;
      IOUtils.quietClose(resultSet, statement, connection);
    }

  }

}
