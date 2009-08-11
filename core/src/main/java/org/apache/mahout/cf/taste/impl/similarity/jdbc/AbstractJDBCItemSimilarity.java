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

package org.apache.mahout.cf.taste.impl.similarity.jdbc;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.cf.taste.impl.model.jdbc.ConnectionPoolDataSource;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;

/**
 * An {@link ItemSimilarity} which draws pre-computed item-item similarities from
 * a database table via JDBC.
 */
public abstract class AbstractJDBCItemSimilarity extends AbstractJDBCComponent implements ItemSimilarity {

  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCItemSimilarity.class);

  static final String DEFAULT_SIMILARITY_TABLE = "taste_item_similarity";
  static final String DEFAULT_ITEM_A_ID_COLUMN = "item_id_a";
  static final String DEFAULT_ITEM_B_ID_COLUMN = "item_id_b";
  static final String DEFAULT_SIMILARITY_COLUMN = "similarity";

  private final DataSource dataSource;
  private final String similarityTable;
  private final String itemAIDColumn;
  private final String itemBIDColumn;
  private final String similarityColumn;
  private final String getItemItemSimilaritySQL;

  protected AbstractJDBCItemSimilarity(DataSource dataSource,
                                       String getItemItemSimilaritySQL) {
    this(dataSource,
         DEFAULT_SIMILARITY_TABLE,
         DEFAULT_ITEM_A_ID_COLUMN,
         DEFAULT_ITEM_B_ID_COLUMN,
         DEFAULT_SIMILARITY_COLUMN,
         getItemItemSimilaritySQL);
  }

  protected AbstractJDBCItemSimilarity(DataSource dataSource,
                                       String similarityTable,
                                       String itemAIDColumn,
                                       String itemBIDColumn,
                                       String similarityColumn,
                                       String getItemItemSimilaritySQL) {
    checkNotNullAndLog("similarityTable", similarityTable);
    checkNotNullAndLog("itemAIDColumn", itemAIDColumn);
    checkNotNullAndLog("itemBIDColumn", itemBIDColumn);
    checkNotNullAndLog("similarityColumn", similarityColumn);

    checkNotNullAndLog("getItemItemSimilaritySQL", getItemItemSimilaritySQL);

    if (!(dataSource instanceof ConnectionPoolDataSource)) {
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections " +
          "to the database itself, or database performance will be severely reduced.");
    }

    this.dataSource = dataSource;
    this.similarityTable = similarityTable;
    this.itemAIDColumn = itemAIDColumn;
    this.itemBIDColumn = itemBIDColumn;
    this.similarityColumn = similarityColumn;
    this.getItemItemSimilaritySQL = getItemItemSimilaritySQL;
  }

  protected String getSimilarityTable() {
    return similarityTable;
  }

  protected String getItemAIDColumn() {
    return itemAIDColumn;
  }

  protected String getItemBIDColumn() {
    return itemBIDColumn;
  }

  protected String getSimilarityColumn() {
    return similarityColumn;
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {

    if (itemID1 == itemID2) {
      return 1.0;
    }
    // Order as smaller - larger
    if (itemID1 > itemID2) {
      long temp = itemID1;
      itemID1 = itemID2;
      itemID2 = temp;
    }

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getItemItemSimilaritySQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, itemID1);
      stmt.setLong(2, itemID2);

      log.debug("Executing SQL query: {}", getItemItemSimilaritySQL);
      rs = stmt.executeQuery();

      if (rs.next()) {
        return rs.getDouble(1);
      } else {
        throw new NoSuchItemException();
      }

    } catch (SQLException sqle) {
      log.warn("Exception while retrieving user", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
