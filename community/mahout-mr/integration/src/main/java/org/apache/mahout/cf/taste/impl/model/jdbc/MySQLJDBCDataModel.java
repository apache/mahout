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

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;

/**
 * <p>
 * A {@link org.apache.mahout.cf.taste.model.JDBCDataModel} backed by a MySQL database and
 * accessed via JDBC. It may work with other JDBC databases. By default, this class
 * assumes that there is a {@link DataSource} available under the JNDI name
 * "jdbc/taste", which gives access to a database with a "taste_preferences" table with the following schema:
 * </p>
 *
 * <table>
 * <tr>
 * <th>user_id</th>
 * <th>item_id</th>
 * <th>preference</th>
 * </tr>
 * <tr>
 * <td>987</td>
 * <td>123</td>
 * <td>0.9</td>
 * </tr>
 * <tr>
 * <td>987</td>
 * <td>456</td>
 * <td>0.1</td>
 * </tr>
 * <tr>
 * <td>654</td>
 * <td>123</td>
 * <td>0.2</td>
 * </tr>
 * <tr>
 * <td>654</td>
 * <td>789</td>
 * <td>0.3</td>
 * </tr>
 * </table>
 *
 * <p>
 * {@code preference} must have a type compatible with the Java {@code float} type.
 * {@code user_id} and {@code item_id} should be compatible with long type (BIGINT). For example,
 * the following command sets up a suitable table in MySQL, complete with primary key and indexes:
 * </p>
 *
 * <p>
 *
 * <pre>
 * CREATE TABLE taste_preferences (
 *   user_id BIGINT NOT NULL,
 *   item_id BIGINT NOT NULL,
 *   preference FLOAT NOT NULL,
 *   PRIMARY KEY (user_id, item_id),
 *   INDEX (user_id),
 *   INDEX (item_id)
 * )
 * </pre>
 *
 * </p>
 *
 * <p>The table may optionally have a {@code timestamp} column whose type is compatible with Java
 * {@code long}.</p>
 *
 * <h3>Performance Notes</h3>
 *
 * <p>
 * See the notes in {@link AbstractJDBCDataModel} regarding using connection pooling. It's pretty vital to
 * performance.
 * </p>
 *
 * <p>
 * Some experimentation suggests that MySQL's InnoDB engine is faster than MyISAM for these kinds of
 * applications. While MyISAM is the default and, I believe, generally considered the lighter-weight and
 * faster of the two engines, my guess is the row-level locking of InnoDB helps here. Your mileage may vary.
 * </p>
 *
 * <p>
 * Here are some key settings that can be tuned for MySQL, and suggested size for a data set of around 1
 * million elements:
 * </p>
 *
 * <ul>
 * <li>innodb_buffer_pool_size=64M</li>
 * <li>myisam_sort_buffer_size=64M</li>
 * <li>query_cache_limit=64M</li>
 * <li>query_cache_min_res_unit=512K</li>
 * <li>query_cache_type=1</li>
 * <li>query_cache_size=64M</li>
 * </ul>
 *
 * <p>
 * Also consider setting some parameters on the MySQL Connector/J driver:
 * </p>
 *
 * <pre>
 * cachePreparedStatements = true
 * cachePrepStmts = true
 * cacheResultSetMetadata = true
 * alwaysSendSetIsolation = false
 * elideSetAutoCommits = true
 * </pre>
 *
 * <p>
 * Thanks to Amila Jayasooriya for contributing MySQL notes above as part of Google Summer of Code 2007.
 * </p>
 */
public class MySQLJDBCDataModel extends AbstractJDBCDataModel {
  
  /**
   * <p>
   * Creates a {@link MySQLJDBCDataModel} using the default {@link DataSource} (named
   * {@link #DEFAULT_DATASOURCE_NAME} and default table/column names.
   * </p>
   * 
   * @throws TasteException
   *           if {@link DataSource} can't be found
   */
  public MySQLJDBCDataModel() throws TasteException {
    this(DEFAULT_DATASOURCE_NAME);
  }
  
  /**
   * <p>
   * Creates a {@link MySQLJDBCDataModel} using the default {@link DataSource} found under the given name, and
   * using default table/column names.
   * </p>
   * 
   * @param dataSourceName
   *          name of {@link DataSource} to look up
   * @throws TasteException
   *           if {@link DataSource} can't be found
   */
  public MySQLJDBCDataModel(String dataSourceName) throws TasteException {
    this(AbstractJDBCComponent.lookupDataSource(dataSourceName),
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }
  
  /**
   * <p>
   * Creates a {@link MySQLJDBCDataModel} using the given {@link DataSource} and default table/column names.
   * </p>
   * 
   * @param dataSource
   *          {@link DataSource} to use
   */
  public MySQLJDBCDataModel(DataSource dataSource) {
    this(dataSource,
         DEFAULT_PREFERENCE_TABLE,
         DEFAULT_USER_ID_COLUMN,
         DEFAULT_ITEM_ID_COLUMN,
         DEFAULT_PREFERENCE_COLUMN,
         DEFAULT_PREFERENCE_TIME_COLUMN);
  }
  
  /**
   * <p>
   * Creates a {@link MySQLJDBCDataModel} using the given {@link DataSource} and default table/column names.
   * </p>
   * 
   * @param dataSource
   *          {@link DataSource} to use
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
  public MySQLJDBCDataModel(DataSource dataSource,
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
            + ") VALUES (?,?,?) ON DUPLICATE KEY UPDATE " + preferenceColumn + "=?",
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
  }
  
  @Override
  protected int getFetchSize() {
    // Need to return this for MySQL Connector/J to make it use streaming mode
    return Integer.MIN_VALUE;
  }
  
}
