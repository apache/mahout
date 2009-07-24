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

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.common.SkippingIterator;
import org.apache.mahout.cf.taste.impl.model.BooleanPrefUser;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;


public abstract class AbstractBooleanPrefJDBCDataModel extends AbstractJDBCDataModel {

  private final String getUserSQL;
  private final String setPreferenceSQL;
  private final String getUsersSQL;
  private final String getPrefsForItemSQL;

  protected AbstractBooleanPrefJDBCDataModel(DataSource dataSource,
                                             String preferenceTable,
                                             String userIDColumn,
                                             String itemIDColumn,
                                             String preferenceColumn,
                                             String getUserSQL,
                                             String getNumItemsSQL,
                                             String getNumUsersSQL,
                                             String setPreferenceSQL,
                                             String removePreferenceSQL,
                                             String getUsersSQL,
                                             String getItemsSQL,
                                             String getPrefsForItemSQL,
                                             String getNumPreferenceForItemSQL,
                                             String getNumPreferenceForItemsSQL) {
    super(dataSource,
        preferenceTable,
        userIDColumn,
        itemIDColumn,
        preferenceColumn,
        getUserSQL,
        getNumItemsSQL,
        getNumUsersSQL,
        setPreferenceSQL,
        removePreferenceSQL,
        getUsersSQL,
        getItemsSQL,
        getPrefsForItemSQL,
        getNumPreferenceForItemSQL,
        getNumPreferenceForItemsSQL);
    this.getUserSQL = getUserSQL;
    this.setPreferenceSQL = setPreferenceSQL;
    this.getUsersSQL = getUsersSQL;
    this.getPrefsForItemSQL = getPrefsForItemSQL;
  }

  /**
   * @throws org.apache.mahout.cf.taste.common.NoSuchUserException
   *          if there is no such user
   */
  @Override
  public User getUser(Comparable<?> id) throws TasteException {

    log.debug("Retrieving user ID '{}'", id);

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    try {
      conn = getDataSource().getConnection();
      stmt = conn.prepareStatement(getUserSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setObject(1, id);

      log.debug("Executing SQL query: {}", getUserSQL);
      rs = stmt.executeQuery();

      FastSet<Comparable<?>> itemIDs = new FastSet<Comparable<?>>();
      while (rs.next()) {
        itemIDs.add((Comparable<?>) rs.getObject(1));
      }

      if (itemIDs.isEmpty()) {
        throw new NoSuchUserException();
      }

      return buildUser(id, itemIDs);

    } catch (SQLException sqle) {
      log.warn("Exception while retrieving user", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }

  }

  @Override
  public Iterable<? extends User> getUsers() throws TasteException {
    log.debug("Retrieving all users...");
    return new IteratorIterable<User>(new ResultSetUserIterator(getDataSource(), getUsersSQL));
  }

  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, double value)
      throws TasteException {
    if (userID == null || itemID == null) {
      throw new IllegalArgumentException("userID or itemID is null");
    }
    if (!Double.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }

    if (log.isDebugEnabled()) {
      log.debug("Setting preference for user '" + userID + "', item '" + itemID);
    }

    Connection conn = null;
    PreparedStatement stmt = null;

    try {
      conn = getDataSource().getConnection();
      stmt = conn.prepareStatement(setPreferenceSQL);
      stmt.setObject(1, userID);
      stmt.setObject(2, itemID);

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
  protected List<? extends Preference> doGetPreferencesForItem(Comparable<?> itemID) throws TasteException {
    log.debug("Retrieving preferences for item ID '{}'", itemID);
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = getDataSource().getConnection();
      stmt = conn.prepareStatement(getPrefsForItemSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setObject(1, itemID);

      log.debug("Executing SQL query: {}", getPrefsForItemSQL);
      rs = stmt.executeQuery();
      List<Preference> prefs = new ArrayList<Preference>();
      while (rs.next()) {
        Comparable<?> userID = (Comparable<?>) rs.getObject(2);
        Preference pref = buildPreference(buildUser(userID, (FastSet<Comparable<?>>) null), itemID);
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

  protected User buildUser(Comparable<?> id, FastSet<Comparable<?>> itemIDs) {
    return new BooleanPrefUser(id, itemIDs);
  }

  protected Preference buildPreference(User user, Comparable<?> itemID) {
    return new BooleanPreference(user, itemID);
  }

  private final class ResultSetUserIterator implements SkippingIterator<User> {

    private final Connection connection;
    private final Statement statement;
    private final ResultSet resultSet;
    private boolean closed;

    private ResultSetUserIterator(DataSource dataSource, String getUsersSQL) throws TasteException {
      try {
        connection = dataSource.getConnection();
        statement = connection.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
        statement.setFetchDirection(ResultSet.FETCH_FORWARD);
        statement.setFetchSize(getFetchSize());
        log.debug("Executing SQL query: {}", getUsersSQL);
        resultSet = statement.executeQuery(getUsersSQL);
        boolean anyResults = resultSet.next();
        if (!anyResults) {
          close();
        }
      } catch (SQLException sqle) {
        close();
        throw new TasteException(sqle);
      }
    }

    @Override
    public boolean hasNext() {
      boolean nextExists = false;
      if (!closed) {
        try {
          if (resultSet.isAfterLast()) {
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

    @Override
    public User next() {

      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      Comparable<?> currentUserID = null;
      FastSet<Comparable<?>> itemIDs = new FastSet<Comparable<?>>();

      try {
        do {
          Comparable<?> userID = (Comparable<?>) resultSet.getObject(2);
          if (currentUserID == null) {
            currentUserID = userID;
          }
          // Did we move on to a new user?
          if (!userID.equals(currentUserID)) {
            break;
          }
          // else add a new preference for the current user
          itemIDs.add((Comparable<?>) resultSet.getObject(1));
        } while (resultSet.next());
      } catch (SQLException sqle) {
        // No good way to handle this since we can't throw an exception
        log.warn("Exception while iterating over users", sqle);
        close();
        throw new NoSuchElementException("Can't retrieve more due to exception: " + sqle);
      }

      return buildUser(currentUserID, itemIDs);
    }

    /**
     * @throws UnsupportedOperationException
     */
    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

    private void close() {
      closed = true;
      IOUtils.quietClose(resultSet, statement, connection);
    }

    @Override
    public void skip(int n) {
      if (n >= 1 && hasNext()) {
        try {
          int distinctUserNamesSeen = 0;
          Object currentUserID = null;
          do {
            Comparable<?> userID = (Comparable<?>) resultSet.getObject(2);
            if (!userID.equals(currentUserID)) {
              distinctUserNamesSeen++;
            }
            currentUserID = userID;
          } while (distinctUserNamesSeen <= n && resultSet.next());
        } catch (SQLException sqle) {
          log.warn("Exception while iterating over users", sqle);
          close();
        }
      }
    }
  }

}