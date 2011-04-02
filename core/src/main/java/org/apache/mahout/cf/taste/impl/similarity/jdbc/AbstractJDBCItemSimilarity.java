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

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.cf.taste.impl.model.jdbc.ConnectionPoolDataSource;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An {@link ItemSimilarity} which draws pre-computed item-item similarities from a database table via JDBC.
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
  private final String getAllSimilarItemIDsSQL;

  protected AbstractJDBCItemSimilarity(DataSource dataSource,
                                       String getItemItemSimilaritySQL,
                                       String getAllSimilarItemIDsSQL) {
    this(dataSource,
         DEFAULT_SIMILARITY_TABLE,
         DEFAULT_ITEM_A_ID_COLUMN,
         DEFAULT_ITEM_B_ID_COLUMN,
         DEFAULT_SIMILARITY_COLUMN,
         getItemItemSimilaritySQL,
         getAllSimilarItemIDsSQL);
  }
  
  protected AbstractJDBCItemSimilarity(DataSource dataSource,
                                       String similarityTable,
                                       String itemAIDColumn,
                                       String itemBIDColumn,
                                       String similarityColumn,
                                       String getItemItemSimilaritySQL,
                                       String getAllSimilarItemIDsSQL) {
    AbstractJDBCComponent.checkNotNullAndLog("similarityTable", similarityTable);
    AbstractJDBCComponent.checkNotNullAndLog("itemAIDColumn", itemAIDColumn);
    AbstractJDBCComponent.checkNotNullAndLog("itemBIDColumn", itemBIDColumn);
    AbstractJDBCComponent.checkNotNullAndLog("similarityColumn", similarityColumn);
    
    AbstractJDBCComponent.checkNotNullAndLog("getItemItemSimilaritySQL", getItemItemSimilaritySQL);
    AbstractJDBCComponent.checkNotNullAndLog("getAllSimilarItemIDsSQL", getAllSimilarItemIDsSQL);

    if (!(dataSource instanceof ConnectionPoolDataSource)) {
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections "
               + "to the database itself, or database performance will be severely reduced.");
    }
    
    this.dataSource = dataSource;
    this.similarityTable = similarityTable;
    this.itemAIDColumn = itemAIDColumn;
    this.itemBIDColumn = itemBIDColumn;
    this.similarityColumn = similarityColumn;
    this.getItemItemSimilaritySQL = getItemItemSimilaritySQL;
    this.getAllSimilarItemIDsSQL = getAllSimilarItemIDsSQL;
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
    Connection conn = null;
    PreparedStatement stmt = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getItemItemSimilaritySQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      return doItemSimilarity(stmt, itemID1, itemID2);
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving similarity", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    double[] result = new double[itemID2s.length];
    Connection conn = null;
    PreparedStatement stmt = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getItemItemSimilaritySQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      for (int i = 0; i < itemID2s.length; i++) {
        result[i] = doItemSimilarity(stmt, itemID1, itemID2s[i]);
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving item similarities", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
    return result;
  }

  @Override
  public long[] allSimilarItemIDs(long itemID) throws TasteException {
    FastIDSet allSimilarItemIDs = new FastIDSet();
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getAllSimilarItemIDsSQL, ResultSet.TYPE_FORWARD_ONLY,
          ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, itemID);
      stmt.setLong(2, itemID);
      rs = stmt.executeQuery();
      while (rs.next()) {
        allSimilarItemIDs.add(rs.getLong(1));
        allSimilarItemIDs.add(rs.getLong(2));
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving all similar itemIDs", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
    allSimilarItemIDs.remove(itemID);
    return allSimilarItemIDs.toArray();
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
  // do nothing
  }

  private double doItemSimilarity(PreparedStatement stmt, long itemID1, long itemID2) throws SQLException {
    // Order as smaller - larger
    if (itemID1 > itemID2) {
      long temp = itemID1;
      itemID1 = itemID2;
      itemID2 = temp;
    }
    stmt.setLong(1, itemID1);
    stmt.setLong(2, itemID2);
    log.debug("Executing SQL query: {}", getItemItemSimilaritySQL);
    ResultSet rs = null;
    try {
      rs = stmt.executeQuery();
      // If not found, perhaps the items exist but have no presence in the table,
      // so NaN is appropriate
      return rs.next() ? rs.getDouble(1) : Double.NaN;
    } finally {
      IOUtils.quietClose(rs);
    }
  }
  
}
