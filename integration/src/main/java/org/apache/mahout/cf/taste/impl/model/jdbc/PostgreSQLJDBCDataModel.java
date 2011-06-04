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

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

/**
 * <p>
 * A {@link org.apache.mahout.cf.taste.model.JDBCDataModel} backed by a PostgreSQL database and
 * accessed via JDBC. It may work with other JDBC databases. By default, this class
 * assumes that there is a {@link javax.sql.DataSource} available under the JNDI name
 * "jdbc/taste", which gives access to a database with a "taste_preferences" table with the following schema:
 * </p>
 *
 * <p>
 *
 * <pre>
 * CREATE TABLE taste_preferences (
 *   user_id BIGINT NOT NULL,
 *   item_id BIGINT NOT NULL,
 *   preference REAL NOT NULL,
 *   PRIMARY KEY (user_id, item_id)
 * )
 * CREATE INDEX taste_preferences_user_id_index ON taste_preferences (user_id);
 * CREATE INDEX taste_preferences_item_id_index ON taste_preferences (item_id);
 * </pre>
 *
 * </p>
 *
 * <p>See {@link MySQLJDBCDataModel} which is largely identical.</p>
 *
 * @see MySQLJDBCDataModel
 */
public class PostgreSQLJDBCDataModel extends AbstractJDBCDataModel {

  private static final Logger log = LoggerFactory.getLogger(PostgreSQLJDBCDataModel.class);
  private static final String POSTGRESQL_DUPLICATE_KEY_STATE = "23505"; // this is brittle...

  private final String updatePreferenceSQL;

  /**
   * <p>
   * Creates a  using the default {@link javax.sql.DataSource} (named
   * {@link #DEFAULT_DATASOURCE_NAME} and default table/column names.
   * </p>
   *
   * @throws org.apache.mahout.cf.taste.common.TasteException
   *           if {@link javax.sql.DataSource} can't be found
   */
  public PostgreSQLJDBCDataModel() throws TasteException {
    this(DEFAULT_DATASOURCE_NAME);
  }

  /**
   * <p>
   * Creates a  using the default {@link javax.sql.DataSource} found under the given name, and
   * using default table/column names.
   * </p>
   *
   * @param dataSourceName
   *          name of {@link javax.sql.DataSource} to look up
   * @throws org.apache.mahout.cf.taste.common.TasteException
   *           if {@link javax.sql.DataSource} can't be found
   */
  public PostgreSQLJDBCDataModel(String dataSourceName) throws TasteException {
    this(AbstractJDBCComponent.lookupDataSource(dataSourceName),
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }

  /**
   * <p>
   * Creates a  using the given {@link javax.sql.DataSource} and default table/column names.
   * </p>
   *
   * @param dataSource
   *          {@link javax.sql.DataSource} to use
   */
  public PostgreSQLJDBCDataModel(DataSource dataSource) {
    this(dataSource,
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }

  /**
   * <p>
   * Creates a  using the given {@link javax.sql.DataSource} and default table/column names.
   * </p>
   *
   * @param dataSource
   *          {@link javax.sql.DataSource} to use
   * @param preferenceTable
   *          name of table containing preference data
   * @param userIDColumn
   *          user ID column name
   * @param itemIDColumn
   *          item ID column name
   * @param preferenceColumn
   *          preference column name
   * @param timestampColumn timestamp column name (may be null)
   */
  public PostgreSQLJDBCDataModel(DataSource dataSource,
                                 String preferenceTable,
                                 String userIDColumn,
                                 String itemIDColumn,
                                 String preferenceColumn,
                                 String timestampColumn) {
    super(dataSource, preferenceTable, userIDColumn, itemIDColumn, preferenceColumn,
        // getPreferenceSQL
        "SELECT " + preferenceColumn + " FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND "
            + itemIDColumn + "=?",
        // getPreferenceTimeSQL
        "SELECT " + timestampColumn + " FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND "
            + itemIDColumn + "=?",
        // getUserSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + ", " + preferenceColumn + " FROM " + preferenceTable
            + " WHERE " + userIDColumn + "=? ORDER BY " + itemIDColumn,
        // getAllUsersSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + ", " + preferenceColumn + " FROM " + preferenceTable
            + " ORDER BY " + userIDColumn + ", " + itemIDColumn,
        // getNumItemsSQL
        "SELECT COUNT(DISTINCT " + itemIDColumn + ") FROM " + preferenceTable,
        // getNumUsersSQL
        "SELECT COUNT(DISTINCT " + userIDColumn + ") FROM " + preferenceTable,
        // setPreferenceSQL
        "INSERT INTO " + preferenceTable + '(' + userIDColumn + ',' + itemIDColumn + ',' + preferenceColumn
            + ") VALUES (?,?,?)",
        // removePreference SQL
        "DELETE FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND " + itemIDColumn + "=?",
        // getUsersSQL
        "SELECT DISTINCT " + userIDColumn + " FROM " + preferenceTable + " ORDER BY " + userIDColumn,
        // getItemsSQL
        "SELECT DISTINCT " + itemIDColumn + " FROM " + preferenceTable + " ORDER BY " + itemIDColumn,
        // getPrefsForItemSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + ", " + preferenceColumn + " FROM " + preferenceTable
            + " WHERE " + itemIDColumn + "=? ORDER BY " + userIDColumn,
        // getNumPreferenceForItemSQL
        "SELECT COUNT(1) FROM " + preferenceTable + " WHERE " + itemIDColumn + "=?",
        // getNumPreferenceForItemsSQL
        "SELECT COUNT(1) FROM " + preferenceTable + " tp1 JOIN " + preferenceTable + " tp2 " + "USING ("
            + userIDColumn + ") WHERE tp1." + itemIDColumn + "=? and tp2." + itemIDColumn + "=?",
        "SELECT MAX(" + preferenceColumn + ") FROM " + preferenceTable,
        "SELECT MIN(" + preferenceColumn + ") FROM " + preferenceTable);

    updatePreferenceSQL = "UPDATE " + preferenceTable + " SET " + preferenceColumn + "=? WHERE " + userIDColumn
        + "=? AND " + itemIDColumn + "=?";
  }

  /**
   * Override since PostgreSQL doesn't have the same non-standard capability that MySQL has, to optionally
   * insert or update in one statement.
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    Preconditions.checkArgument(!Float.isNaN(value), "NaN value");

    log.debug("Setting preference for user {}, item {}", userID, itemID);

    String setPreferenceSQL = getSetPreferenceSQL();

    Connection conn = null;
    PreparedStatement stmt1 = null;
    PreparedStatement stmt2 = null;
    try {
      conn = getDataSource().getConnection();

      stmt1 = conn.prepareStatement(setPreferenceSQL);
      setLongParameter(stmt1, 1, userID);
      setLongParameter(stmt1, 2, itemID);
      stmt1.setDouble(3, value);

      log.debug("Executing SQL update: {}", setPreferenceSQL);
      try {
        stmt1.executeUpdate();
      } catch (SQLException sqle) {
        if (!POSTGRESQL_DUPLICATE_KEY_STATE.equals(sqle.getSQLState())) {
          throw sqle;
        }
      }

      // Continue with update; just found the key already exists

      stmt2 = conn.prepareStatement(updatePreferenceSQL);
      stmt2.setDouble(1, value);
      setLongParameter(stmt2, 2, userID);
      setLongParameter(stmt2, 3, itemID);

      log.debug("Executing SQL update: {}", updatePreferenceSQL);
      stmt2.executeUpdate();

    } catch (SQLException sqle) {
      log.warn("Exception while setting preference", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt1, null);
      IOUtils.quietClose(null, stmt2, null);
      IOUtils.quietClose(null, null, conn);
    }
  }
  
}
