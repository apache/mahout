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
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * <p>
 * See also {@link org.apache.mahout.cf.taste.impl.model.jdbc.SQL92JDBCDataModel} --
 * same except deals with a table without preference info:
 * </p>
 *
 * <p>
 *
 * <pre>
 * CREATE TABLE taste_preferences (
 *   user_id BIGINT NOT NULL,
 *   item_id BIGINT NOT NULL,
 *   PRIMARY KEY (user_id, item_id)
 * );
 * CREATE INDEX taste_preferences_user_id_index ON taste_preferences (user_id);
 * CREATE INDEX taste_preferences_item_id_index ON taste_preferences (item_id);
 * </pre>
 *
 * </p>
 *
 * @see SQL92JDBCDataModel
 */
public class SQL92BooleanPrefJDBCDataModel extends AbstractBooleanPrefJDBCDataModel {

  private static final Logger log = LoggerFactory.getLogger(SQL92BooleanPrefJDBCDataModel.class);

  private final String verifyPreferenceSQL;

  /**
   * <p>
   * Creates a  using the default {@link javax.sql.DataSource} (named
   * {@link #DEFAULT_DATASOURCE_NAME} and default table/column names.
   * </p>
   *
   * @throws org.apache.mahout.cf.taste.common.TasteException
   *           if {@link javax.sql.DataSource} can't be found
   */
  public SQL92BooleanPrefJDBCDataModel() throws TasteException {
    this(DEFAULT_DATASOURCE_NAME);
  }

  /**
   * <p>
   * Creates a  using the default {@link javax.sql.DataSource} found
   * under the given name, and using default table/column names.
   * </p>
   *
   * @param dataSourceName
   *          name of {@link javax.sql.DataSource} to look up
   * @throws org.apache.mahout.cf.taste.common.TasteException
   *           if {@link javax.sql.DataSource} can't be found
   */
  public SQL92BooleanPrefJDBCDataModel(String dataSourceName) throws TasteException {
    this(AbstractJDBCComponent.lookupDataSource(dataSourceName),
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }

  /**
   * <p>
   * Creates a  using the given {@link javax.sql.DataSource} and default
   * table/column names.
   * </p>
   *
   * @param dataSource
   *          {@link javax.sql.DataSource} to use
   */
  public SQL92BooleanPrefJDBCDataModel(DataSource dataSource) {
    this(dataSource,
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }

  /**
   * <p>
   * Creates a  using the given {@link javax.sql.DataSource} and default
   * table/column names.
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
   * @param timestampColumn timestamp column name (may be null)
   */
  public SQL92BooleanPrefJDBCDataModel(DataSource dataSource,
                                            String preferenceTable,
                                            String userIDColumn,
                                            String itemIDColumn,
                                            String timestampColumn) {
    super(dataSource, preferenceTable, userIDColumn, itemIDColumn,
        NO_SUCH_COLUMN,
        // getPreferenceSQL
        "SELECT 1 FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND " + itemIDColumn + "=?",
        // getPreferenceTimeSQL
        "SELECT " + timestampColumn + " FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND "
            + itemIDColumn + "=?",
        // getUserSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + " FROM " + preferenceTable + " WHERE "
            + userIDColumn + "=?",
        // getAllUsersSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + " FROM " + preferenceTable + " ORDER BY "
            + userIDColumn,
        // getNumItemsSQL
        "SELECT COUNT(DISTINCT " + itemIDColumn + ") FROM " + preferenceTable,
        // getNumUsersSQL
        "SELECT COUNT(DISTINCT " + userIDColumn + ") FROM " + preferenceTable,
        // setPreferenceSQL
        "INSERT INTO " + preferenceTable + '(' + userIDColumn + ',' + itemIDColumn + ") VALUES (?,?)",
        // removePreference SQL
        "DELETE FROM " + preferenceTable + " WHERE " + userIDColumn + "=? AND " + itemIDColumn + "=?",
        // getUsersSQL
        "SELECT DISTINCT " + userIDColumn + " FROM " + preferenceTable + " ORDER BY " + userIDColumn,
        // getItemsSQL
        "SELECT DISTINCT " + itemIDColumn + " FROM " + preferenceTable + " ORDER BY " + itemIDColumn,
        // getPrefsForItemSQL
        "SELECT DISTINCT " + userIDColumn + ", " + itemIDColumn + " FROM " + preferenceTable + " WHERE "
            + itemIDColumn + "=? ORDER BY " + userIDColumn,
        // getNumPreferenceForItemSQL
        "SELECT COUNT(1) FROM " + preferenceTable + " WHERE " + itemIDColumn + "=?",
        // getNumPreferenceForItemsSQL
        "SELECT COUNT(1) FROM " + preferenceTable + " tp1 JOIN " + preferenceTable + " tp2 " + "USING ("
            + userIDColumn + ") WHERE tp1." + itemIDColumn + "=? and tp2." + itemIDColumn + "=?",
        // getMaxPreferenceSQL
        "SELECT 1.0",
        // getMinPreferenceSQL
        "SELECT 1.0");

    verifyPreferenceSQL = "SELECT 1 FROM " + preferenceTable + " WHERE " + userIDColumn
        + "=? AND " + itemIDColumn + "=?";
  }

  protected String getVerifyPreferenceSQL() {
    return verifyPreferenceSQL;
  }

  /**
   * Override since PostgreSQL doesn't have the same non-standard capability that MySQL has, to optionally
   * ignore an insert that fails since the row exists already.
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    Preconditions.checkArgument(!Float.isNaN(value), "NaN value");
    log.debug("Setting preference for user {}, item {}", userID, itemID);

    String setPreferenceSQL = getSetPreferenceSQL();

    Connection conn = null;
    PreparedStatement stmt1 = null;
    PreparedStatement stmt2 = null;
    ResultSet rs = null;
    try {
      conn = getDataSource().getConnection();

      stmt1 = conn.prepareStatement(verifyPreferenceSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      setLongParameter(stmt1, 1, userID);
      setLongParameter(stmt1, 2, itemID);
      rs = stmt1.executeQuery();

      // test if the record exists already.
      if (!rs.first()) {
        stmt2 = conn.prepareStatement(setPreferenceSQL);
        setLongParameter(stmt2, 1, userID);
        setLongParameter(stmt2, 2, itemID);
        stmt2.setDouble(3, value);

        log.debug("Executing SQL update: {}", setPreferenceSQL);
        stmt2.executeUpdate();
      }
    } catch (SQLException sqle) {
      log.warn("Exception while setting preference", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs);
      IOUtils.quietClose(stmt1);
      IOUtils.quietClose(stmt2);
      IOUtils.quietClose(conn);
    }
  }

}
