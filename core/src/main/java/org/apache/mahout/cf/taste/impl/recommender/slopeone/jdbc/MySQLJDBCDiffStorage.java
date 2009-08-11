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

package org.apache.mahout.cf.taste.impl.recommender.slopeone.jdbc;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.jdbc.AbstractJDBCDataModel;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;

/**
 * <p>MySQL-specific implementation. Should be used in conjunction with a {@link MySQLJDBCDataModel}. This
 * implementation stores item-item diffs in a MySQL database and encapsulates some other slope-one-specific operations
 * that are needed on the preference data in the database. It assumes the database has a schema like:</p>
 *
 * <table> <tr><th>item_id_a</th><th>item_id_b</th><th>average_diff</th><th>count</th></tr>
 * <tr><td>123</td><td>234</td><td>0.5</td><td>5</td></tr> <tr><td>123</td><td>789</td><td>-1.33</td><td>3</td></tr>
 * <tr><td>234</td><td>789</td><td>2.1</td><td>1</td></tr> </table>
 *
 * <p><code>item_id_a</code> and <code>item_id_b</code> should have types compatible with the long
 * primitive type. <code>average_diff</code> must be compatible with <code>float</code> and
 * <code>count</code> must be compatible with <code>int</code>.</p>
 *
 * <p>The following command sets up a suitable table in MySQL:</p>
 *
 * <pre>
 * CREATE TABLE taste_slopeone_diffs (
 *   item_id_a BIGINT NOT NULL,
 *   item_id_b BIGINT NOT NULL,
 *   average_diff FLOAT NOT NULL,
 *   count INT NOT NULL,
 *   PRIMARY KEY (item_id_a, item_id_b),
 *   INDEX (item_id_a),
 *   INDEX (item_id_b)
 * )
 * </pre>
 */
public final class MySQLJDBCDiffStorage extends AbstractJDBCDiffStorage {

  private static final int DEFAULT_MIN_DIFF_COUNT = 2;

  public MySQLJDBCDiffStorage(AbstractJDBCDataModel dataModel) throws TasteException {
    this(dataModel,
        DEFAULT_DIFF_TABLE,
        DEFAULT_ITEM_A_COLUMN,
        DEFAULT_ITEM_B_COLUMN,
        DEFAULT_COUNT_COLUMN,
        DEFAULT_AVERAGE_DIFF_COLUMN,
        DEFAULT_MIN_DIFF_COUNT);
  }

  public MySQLJDBCDiffStorage(AbstractJDBCDataModel dataModel,
                              String diffsTable,
                              String itemIDAColumn,
                              String itemIDBColumn,
                              String countColumn,
                              String avgColumn,
                              int minDiffCount) throws TasteException {
    super(dataModel,
        // getDiffSQL
        "SELECT " + countColumn + ", " + avgColumn + " FROM " + diffsTable +
            " WHERE " + itemIDAColumn + "=? AND " + itemIDBColumn + "=? UNION " +
            "SELECT " + countColumn + ", " + avgColumn + " FROM " + diffsTable +
            " WHERE " + itemIDAColumn + "=? AND " + itemIDBColumn + "=?",
        // getDiffsSQL
        "SELECT " + countColumn + ", " + avgColumn + ", " + itemIDAColumn + " FROM " + diffsTable + ", " +
            dataModel.getPreferenceTable() + " WHERE " + itemIDBColumn + "=? AND " + itemIDAColumn + " = " +
            dataModel.getItemIDColumn() + " AND " + dataModel.getUserIDColumn() + "=? ORDER BY " + itemIDAColumn,
        // getAverageItemPrefSQL
        "SELECT COUNT(1), AVG(" + dataModel.getPreferenceColumn() + ") FROM " + dataModel.getPreferenceTable() +
            " WHERE " + dataModel.getItemIDColumn() + "=?",
        // updateDiffSQLs
        new String[]{
            "UPDATE " + diffsTable + " SET " + avgColumn + " = " + avgColumn + " - (? / " + countColumn +
                ") WHERE " + itemIDAColumn + "=?",
            "UPDATE " + diffsTable + " SET " + avgColumn + " = " + avgColumn + " + (? / " + countColumn +
                ") WHERE " + itemIDBColumn + "=?"
        },
        // removeDiffSQL
        new String[]{
            "UPDATE " + diffsTable + " SET " + countColumn + " = " + countColumn + "-1, " +
                avgColumn + " = " + avgColumn + " * ((" + countColumn + " + 1) / CAST(" + countColumn +
                " AS DECIMAL)) + ? / CAST(" + countColumn + " AS DECIMAL) WHERE " + itemIDAColumn + "=?",
            "UPDATE " + diffsTable + " SET " + countColumn + " = " + countColumn + "-1, " +
                avgColumn + " = " + avgColumn + " * ((" + countColumn + " + 1) / CAST(" + countColumn +
                " AS DECIMAL)) - ? / CAST(" + countColumn + " AS DECIMAL) WHERE " + itemIDBColumn + "=?"
        },
        // getRecommendableItemsSQL
        "SELECT id FROM " +
            "(SELECT " + itemIDAColumn + " AS id FROM " + diffsTable + ", " + dataModel.getPreferenceTable() +
            " WHERE " + itemIDBColumn + " = " + dataModel.getItemIDColumn() +
            " AND " + dataModel.getUserIDColumn() + "=? UNION DISTINCT" +
            " SELECT " + itemIDBColumn + " AS id FROM " + diffsTable + ", " + dataModel.getPreferenceTable() +
            " WHERE " + itemIDAColumn + " = " + dataModel.getItemIDColumn() +
            " AND " + dataModel.getUserIDColumn() +"=?) " +
            "possible_item_ids WHERE id NOT IN (SELECT " + dataModel.getItemIDColumn() + " FROM " +
            dataModel.getPreferenceTable() + " WHERE " + dataModel.getUserIDColumn() + "=?)",
        // deleteDiffsSQL
        "TRUNCATE " + diffsTable,
        // createDiffsSQL
        "INSERT INTO " + diffsTable + " (" + itemIDAColumn + ", " + itemIDBColumn + ", " + avgColumn +
            ", " + countColumn + ") SELECT prefsA." + dataModel.getItemIDColumn() + ", prefsB." +
            dataModel.getItemIDColumn() + ',' +" AVG(prefsB." + dataModel.getPreferenceColumn() +
            " - prefsA." + dataModel.getPreferenceColumn() + ")," + " COUNT(1) AS count FROM " +
            dataModel.getPreferenceTable() + " prefsA, " + dataModel.getPreferenceTable() + " prefsB WHERE prefsA." +
            dataModel.getUserIDColumn() + " = prefsB." + dataModel.getUserIDColumn() +
            " AND prefsA." + dataModel.getItemIDColumn() + " < prefsB." +
            dataModel.getItemIDColumn() + ' ' + " GROUP BY prefsA." + dataModel.getItemIDColumn() +
            ", prefsB." + dataModel.getItemIDColumn() + " HAVING count >=?",
        // diffsExistSQL
        "SELECT COUNT(1) FROM " + diffsTable,
        minDiffCount);
  }

  /**
   * @see MySQLJDBCDataModel#getFetchSize() 
   */
  @Override
  protected int getFetchSize() {
    return Integer.MIN_VALUE;
  }

}