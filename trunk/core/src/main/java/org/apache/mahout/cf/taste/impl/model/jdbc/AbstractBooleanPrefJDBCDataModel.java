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

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.BooleanPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public abstract class AbstractBooleanPrefJDBCDataModel extends AbstractJDBCDataModel {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractBooleanPrefJDBCDataModel.class);

  static final String NO_SUCH_COLUMN = "NO_SUCH_COLUMN";

  private final String setPreferenceSQL;
  
  protected AbstractBooleanPrefJDBCDataModel(DataSource dataSource,
                                             String preferenceTable,
                                             String userIDColumn,
                                             String itemIDColumn,
                                             String preferenceColumn,
                                             String getPreferenceSQL,
                                             String getPreferenceTimeSQL,
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
                                             String getNumPreferenceForItemsSQL,
                                             String getMaxPreferenceSQL,
                                             String getMinPreferenceSQL) {
    super(dataSource,
          preferenceTable,
          userIDColumn,
          itemIDColumn,
          preferenceColumn,
          getPreferenceSQL,
          getPreferenceTimeSQL,
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
          getNumPreferenceForItemsSQL,
          getMaxPreferenceSQL,
          getMinPreferenceSQL);
    this.setPreferenceSQL = setPreferenceSQL;
  }
  
  @Override
  protected Preference buildPreference(ResultSet rs) throws SQLException {
    return new BooleanPreference(getLongColumn(rs, 1), getLongColumn(rs, 2));
  }

  @Override
  String getSetPreferenceSQL() {
    return setPreferenceSQL;
  }
  
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    Preconditions.checkArgument(!Float.isNaN(value), "NaN value");
    log.debug("Setting preference for user {}, item {}", userID, itemID);
    
    Connection conn = null;
    PreparedStatement stmt = null;
    
    try {
      conn = getDataSource().getConnection();
      stmt = conn.prepareStatement(setPreferenceSQL);
      setLongParameter(stmt, 1, userID);
      setLongParameter(stmt, 2, itemID);
      
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
  public boolean hasPreferenceValues() {
    return false;
  }

  @Override
  public float getMaxPreference() {
    return 1.0f;
  }

  @Override
  public float getMinPreference() {
    return 1.0f;
  }
  
}
