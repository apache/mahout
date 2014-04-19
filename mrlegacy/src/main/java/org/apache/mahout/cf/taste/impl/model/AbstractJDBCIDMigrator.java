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

package org.apache.mahout.cf.taste.impl.model;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.UpdatableIDMigrator;
import org.apache.mahout.common.IOUtils;

/**
 * Implementation which stores the reverse long-to-String mapping in a database. Subclasses can override and
 * configure the class to operate with particular databases by supplying appropriate SQL statements to the
 * constructor.
 */
public abstract class AbstractJDBCIDMigrator extends AbstractIDMigrator implements UpdatableIDMigrator {
  
  public static final String DEFAULT_MAPPING_TABLE = "taste_id_mapping";
  public static final String DEFAULT_LONG_ID_COLUMN = "long_id";
  public static final String DEFAULT_STRING_ID_COLUMN = "string_id";
  
  private final DataSource dataSource;
  private final String getStringIDSQL;
  private final String storeMappingSQL;
  
  /**
   * @param getStringIDSQL
   *          SQL statement which selects one column, the String ID, from a mapping table. The statement
   *          should take one long parameter.
   * @param storeMappingSQL
   *          SQL statement which saves a mapping from long to String. It should take two parameters, a long
   *          and a String.
   */
  protected AbstractJDBCIDMigrator(DataSource dataSource, String getStringIDSQL, String storeMappingSQL) {
    this.dataSource = dataSource;
    this.getStringIDSQL = getStringIDSQL;
    this.storeMappingSQL = storeMappingSQL;
  }
  
  @Override
  public final void storeMapping(long longID, String stringID) throws TasteException {
    Connection conn = null;
    PreparedStatement stmt = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(storeMappingSQL);
      stmt.setLong(1, longID);
      stmt.setString(2, stringID);
      stmt.executeUpdate();
    } catch (SQLException sqle) {
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(null, stmt, conn);
    }
  }
  
  @Override
  public final String toStringID(long longID) throws TasteException {
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;
    try {
      conn = dataSource.getConnection();
      stmt = conn.prepareStatement(getStringIDSQL, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
      stmt.setFetchSize(1);
      stmt.setLong(1, longID);
      rs = stmt.executeQuery();
      if (rs.next()) {
        return rs.getString(1);
      } else {
        return null;
      }
    } catch (SQLException sqle) {
      throw new TasteException(sqle);
    } finally {
      IOUtils.quietClose(rs, stmt, conn);
    }
  }

  @Override
  public void initialize(Iterable<String> stringIDs) throws TasteException {
    for (String stringID : stringIDs) {
      storeMapping(toLongID(stringID), stringID);
    }
  }

}
