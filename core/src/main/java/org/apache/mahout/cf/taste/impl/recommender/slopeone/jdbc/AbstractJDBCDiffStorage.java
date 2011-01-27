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

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Collection;
import java.util.concurrent.Callable;

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FixedRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FixedRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * A {@link DiffStorage} which stores diffs in a database. Database-specific implementations subclass this
 * abstract class. Note that this implementation has a fairly particular dependence on the
 * {@link org.apache.mahout.cf.taste.model.DataModel} used; it needs a {@link JDBCDataModel} attached to the
 * same database since its efficient operation depends on accessing preference data in the database directly.
 * </p>
 */
public abstract class AbstractJDBCDiffStorage extends AbstractJDBCComponent implements DiffStorage {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCDiffStorage.class);
  
  public static final String DEFAULT_DIFF_TABLE = "taste_slopeone_diffs";
  public static final String DEFAULT_ITEM_A_COLUMN = "item_id_a";
  public static final String DEFAULT_ITEM_B_COLUMN = "item_id_b";
  public static final String DEFAULT_COUNT_COLUMN = "count";
  public static final String DEFAULT_AVERAGE_DIFF_COLUMN = "average_diff";
  public static final String DEFAULT_STDEV_COLUMN = "standard_deviation";

  private final JDBCDataModel dataModel;
  private final DataSource dataSource;
  private final String getDiffSQL;
  private final String getDiffsSQL;
  private final String getAverageItemPrefSQL;
  private final String getDiffsAffectedByUserSQL;
  private final String[] updateDiffSQLs;
  private final String updateOneDiffSQL;
  private final String addDiffSQL;
  private final String removeDiffSQL;
  private final String getRecommendableItemsSQL;
  private final String deleteDiffsSQL;
  private final String createDiffsSQL;
  private final String diffsExistSQL;
  private final int minDiffCount;
  private final RefreshHelper refreshHelper;
  
  protected AbstractJDBCDiffStorage(JDBCDataModel dataModel,
                                    String getDiffSQL,
                                    String getDiffsSQL,
                                    String getAverageItemPrefSQL,
                                    String getDiffsAffectedByUserSQL,
                                    String[] updateDiffSQLs,
                                    String updateOneDiffSQL,
                                    String addDiffSQL,
                                    String removeDiffSQL,
                                    String getRecommendableItemsSQL,
                                    String deleteDiffsSQL,
                                    String createDiffsSQL,
                                    String diffsExistSQL,
                                    int minDiffCount) throws TasteException {
    
    AbstractJDBCComponent.checkNotNullAndLog("dataModel", dataModel);
    AbstractJDBCComponent.checkNotNullAndLog("getDiffSQL", getDiffSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getDiffsSQL", getDiffsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getAverageItemPrefSQL", getAverageItemPrefSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getDiffsAffectedByUserSQL", getDiffsAffectedByUserSQL);
    AbstractJDBCComponent.checkNotNullAndLog("updateDiffSQLs", updateDiffSQLs);
    AbstractJDBCComponent.checkNotNullAndLog("updateOneDiffSQL", updateOneDiffSQL);
    AbstractJDBCComponent.checkNotNullAndLog("addDiffSQL", addDiffSQL);
    AbstractJDBCComponent.checkNotNullAndLog("removeDiffSQL", removeDiffSQL);
    AbstractJDBCComponent.checkNotNullAndLog("getRecommendableItemsSQL", getRecommendableItemsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("deleteDiffsSQL", deleteDiffsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("createDiffsSQL", createDiffsSQL);
    AbstractJDBCComponent.checkNotNullAndLog("diffsExistSQL", diffsExistSQL);

    Preconditions.checkArgument(minDiffCount >= 0, "minDiffCount is not positive");

    this.dataModel = dataModel;
    this.dataSource = dataModel.getDataSource();
    this.getDiffSQL = getDiffSQL;
    this.getDiffsSQL = getDiffsSQL;
    this.getAverageItemPrefSQL = getAverageItemPrefSQL;
    this.getDiffsAffectedByUserSQL = getDiffsAffectedByUserSQL;
    this.updateDiffSQLs = updateDiffSQLs;
    this.updateOneDiffSQL = updateOneDiffSQL;
    this.addDiffSQL = addDiffSQL;
    this.removeDiffSQL = removeDiffSQL;
    this.getRecommendableItemsSQL = getRecommendableItemsSQL;
    this.deleteDiffsSQL = deleteDiffsSQL;
    this.createDiffsSQL = createDiffsSQL;
    this.diffsExistSQL = diffsExistSQL;
    this.minDiffCount = minDiffCount;
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildAverageDiffs();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    if (isDiffsExist()) {
      log.info("Diffs already exist in database; using them instead of recomputing");
    } else {
      log.info("No diffs exist in database; recomputing...");
      buildAverageDiffs();
    }
  }
  
  @Override
  public RunningAverage getDiff(long itemID1, long itemID2) throws TasteException {

    boolean flipped = itemID1 > itemID2;
    if (flipped) {
      long temp = itemID1;
      itemID1 = itemID2;
      itemID2 = temp;
    }

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getDiffSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, itemID1);
      stmt.setLong(2, itemID2);
      log.debug("Executing SQL query: {}", getDiffSQL);
      rs = stmt.executeQuery();
      if (rs.next()) {
        double average = rs.getDouble(2);
        if (flipped) {
          average = -average;
        }
        return new FixedRunningAverageAndStdDev(average, rs.getDouble(3), rs.getInt(1));
      } else {
        return null;
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }
  
  @Override
  public RunningAverage[] getDiffs(long userID, long itemID, PreferenceArray prefs) throws TasteException {
    int size = prefs.length();
    RunningAverage[] result = new RunningAverage[size];
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getDiffsSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, itemID);
      stmt.setLong(2, userID);
      log.debug("Executing SQL query: {}", getDiffsSQL);
      rs = stmt.executeQuery();
      // We should have up to one result for each Preference in prefs
      // They are both ordered by item. Step through and create a RunningAverage[]
      // with nulls for Preferences that have no corresponding result row
      int i = 0;
      while (rs.next()) {
        long nextResultItemID = rs.getLong(4);
        while (i < size && prefs.getItemID(i) != nextResultItemID) {
          i++;
          // result[i] is null for these values of i
        }
        if (i == size) {
          break;
        }
        result[i] = new FixedRunningAverageAndStdDev(rs.getDouble(2), rs.getDouble(3), rs.getInt(1));
        i++;
      }
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
    return result;
  }
  
  @Override
  public RunningAverage getAverageItemPref(long itemID) throws TasteException {
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getAverageItemPrefSQL, ResultSet.TYPE_FORWARD_ONLY,
        ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, itemID);
      log.debug("Executing SQL query: {}", getAverageItemPrefSQL);
      rs = stmt.executeQuery();
      if (rs.next()) {
        int count = rs.getInt(1);
        if (count > 0) {
          return new FixedRunningAverage(rs.getDouble(2), count);
        }
      }
      return null;
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving average item pref", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }


  @Override
  public void addItemPref(long userID, long itemID, float prefValue) throws TasteException {

    PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
    FastIDSet unupdatedItemIDs = new FastIDSet();
    for (long anItemID : prefs.getIDs()) {
      unupdatedItemIDs.add(anItemID);
    }

    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getDiffsAffectedByUserSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, userID);
      log.debug("Executing SQL query: {}", getDiffsAffectedByUserSQL);
      rs = stmt.executeQuery();

      while (rs.next()) {
        int count = rs.getInt(1);
        float average = rs.getFloat(2);
        long itemIDA = rs.getLong(3);
        long itemIDB = rs.getLong(4);
        float currentOtherPrefValue = rs.getFloat(5);
        float prefDelta;
        long otherItemID;
        if (itemID == itemIDA) {
          prefDelta = currentOtherPrefValue - prefValue;
          otherItemID = itemIDB;
        } else {
          prefDelta = prefValue - currentOtherPrefValue;
          otherItemID = itemIDA;
        }
        float newAverage = (average * count + prefDelta) / (count + 1);
        updateOneDiff(conn, count + 1, newAverage, itemIDA, itemIDB);
        unupdatedItemIDs.remove(otherItemID);
      }

    } catch (SQLException sqle) {
      log.warn("Exception while adding item diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }

    // Catch antyhing that wasn't already covered in the diff table
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(addDiffSQL);
      for (long unupdatedItemID : unupdatedItemIDs) {
        if (unupdatedItemID < itemID) {
          stmt.setLong(1, unupdatedItemID);
          stmt.setLong(2, itemID);
          stmt.setFloat(3, prefValue);
        } else {
          stmt.setLong(1, itemID);
          stmt.setLong(2, unupdatedItemID);
          stmt.setFloat(3, -prefValue);
        }
        log.debug("Executing SQL query: {}", getDiffsAffectedByUserSQL);
        stmt.executeUpdate();
      }
    } catch (SQLException sqle) {
      log.warn("Exception while adding item diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }

  private void updateOneDiff(Connection conn, int newCount, float newAverage, long itemIDA, long itemIDB)
    throws SQLException {
    PreparedStatement stmt = conn.prepareStatement(updateOneDiffSQL);
    try {
      stmt.setInt(1, newCount);
      stmt.setFloat(2, newAverage);
      stmt.setLong(3, itemIDA);
      stmt.setLong(4, itemIDB);
      log.debug("Executing SQL update: {}", updateOneDiffSQL);
      stmt.executeUpdate();
    } finally {
      IOUtils.quietClose(stmt);
    }
  }

  /**
   * Note that this implementation does <em>not</em> update standard deviations. This would
   * be expensive relative to the value of slightly adjusting these values, which are merely
   * used as weighted. Rebuilding the diffs table will update standard deviations.
   */
  @Override
  public void updateItemPref(long itemID, float prefDelta) throws TasteException {
    Connection conn = null;
    try {
      conn = dataSource.getConnection();
      doPartialUpdate(updateDiffSQLs[0], itemID, prefDelta, conn);
      doPartialUpdate(updateDiffSQLs[1], itemID, prefDelta, conn);
    } catch (SQLException sqle) {
      log.warn("Exception while updating item diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(conn);
    }
  }

  @Override
  public void removeItemPref(long userID, long itemID, float prefValue) throws TasteException {
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getDiffsAffectedByUserSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, userID);
      log.debug("Executing SQL query: {}", getDiffsAffectedByUserSQL);
      rs = stmt.executeQuery();

      while (rs.next()) {
        int count = rs.getInt(1);
        long itemIDA = rs.getLong(3);
        long itemIDB = rs.getLong(4);
        if (count == minDiffCount) {
          // going to remove the diff
          removeOneDiff(conn, itemIDA, itemIDB);
        } else {
          float average = rs.getFloat(2);
          float currentOtherPrefValue = rs.getFloat(5);
          float prefDelta;
          if (itemID == itemIDA) {
            prefDelta = currentOtherPrefValue - prefValue;
          } else {
            prefDelta = prefValue - currentOtherPrefValue;
          }
          float newAverage = (average * count - prefDelta) / (count - 1);
          updateOneDiff(conn, count - 1, newAverage, itemIDA, itemIDB);
        }
      }
    } catch (SQLException sqle) {
      log.warn("Exception while removing item diff", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  private void removeOneDiff(Connection conn, long itemIDA, long itemIDB)
    throws SQLException {
    PreparedStatement stmt = conn.prepareStatement(removeDiffSQL);
    try {
      stmt.setLong(1, itemIDA);
      stmt.setLong(2, itemIDB);
      log.debug("Executing SQL update: {}", removeDiffSQL);
      stmt.executeUpdate();
    } finally {
      IOUtils.quietClose(stmt);
    }
  }
  
  private static void doPartialUpdate(String sql, long itemID, double prefDelta, Connection conn) throws SQLException {
    PreparedStatement stmt = conn.prepareStatement(sql);
    try {
      stmt.setDouble(1, prefDelta);
      stmt.setLong(2, itemID);
      log.debug("Executing SQL update: {}", sql);
      stmt.executeUpdate();
    } finally {
      IOUtils.quietClose(stmt);
    }
  }
  
  @Override
  public FastIDSet getRecommendableItemIDs(long userID) throws TasteException {
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getRecommendableItemsSQL, ResultSet.TYPE_FORWARD_ONLY,
        ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      stmt.setLong(1, userID);
      stmt.setLong(2, userID);
      stmt.setLong(3, userID);
      log.debug("Executing SQL query: {}", getRecommendableItemsSQL);
      rs = stmt.executeQuery();
      FastIDSet itemIDs = new FastIDSet();
      while (rs.next()) {
        itemIDs.add(rs.getLong(1));
      }
      return itemIDs;
    } catch (SQLException sqle) {
      log.warn("Exception while retrieving recommendable items", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }
  
  private void buildAverageDiffs() throws TasteException {
    Connection conn = null;
    try {
      conn = dataSource.getConnection();
      PreparedStatement stmt = null;
      try {
        stmt = conn.prepareStatement(deleteDiffsSQL);
        log.debug("Executing SQL update: {}", deleteDiffsSQL);
        stmt.executeUpdate();
      } finally {
        IOUtils.quietClose(stmt);
      }
      try {
        stmt = conn.prepareStatement(createDiffsSQL);
        stmt.setInt(1, minDiffCount);
        log.debug("Executing SQL update: {}", createDiffsSQL);
        stmt.executeUpdate();
      } finally {
        IOUtils.quietClose(stmt);
      }
    } catch (SQLException sqle) {
      log.warn("Exception while updating/deleting diffs", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(conn);
    }
  }
  
  private boolean isDiffsExist() throws TasteException {
    Connection conn = null;
    Statement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.createStatement(ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(getFetchSize());
      log.debug("Executing SQL query: {}", diffsExistSQL);
      rs = stmt.executeQuery(diffsExistSQL);
      rs.next();
      return rs.getInt(1) > 0;
    } catch (SQLException sqle) {
      log.warn("Exception while deleting diffs", sqle);
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
}
