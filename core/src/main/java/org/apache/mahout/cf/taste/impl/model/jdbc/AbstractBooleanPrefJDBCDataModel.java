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
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public abstract class AbstractBooleanPrefJDBCDataModel extends AbstractJDBCDataModel {

  private static final Logger log = LoggerFactory.getLogger(AbstractBooleanPrefJDBCDataModel.class);

  private final String setPreferenceSQL;

  protected AbstractBooleanPrefJDBCDataModel(DataSource dataSource,
                                             String preferenceTable,
                                             String userIDColumn,
                                             String itemIDColumn,
                                             String preferenceColumn,
                                             String getPreferenceSQL,
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
                                             String getNumPreferenceForItemsSQL) {
    super(dataSource,
        preferenceTable,
        userIDColumn,
        itemIDColumn,
        preferenceColumn,
        getPreferenceSQL,
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
        getNumPreferenceForItemsSQL);
    this.setPreferenceSQL = setPreferenceSQL;
  }

  @Override
  protected Preference buildPreference(ResultSet rs) throws SQLException {
    return new BooleanPreference(rs.getLong(1), rs.getLong(2));
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    if (!Float.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }

    log.debug("Setting preference for user {}, item {}", userID, itemID);

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

}