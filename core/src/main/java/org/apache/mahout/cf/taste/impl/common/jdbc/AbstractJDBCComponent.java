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

package org.apache.mahout.cf.taste.impl.common.jdbc;

import org.apache.mahout.cf.taste.common.TasteException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.naming.Context;
import javax.naming.InitialContext;
import javax.naming.NamingException;
import javax.sql.DataSource;
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * A helper class with common elements for several JDBC-related components.
 */
public abstract class AbstractJDBCComponent {

  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCComponent.class);

  protected static final String DEFAULT_DATASOURCE_NAME = "jdbc/taste";
  private static final int DEFAULT_FETCH_SIZE = 1000; // A max, "big" number of rows to buffer at once

  protected static void checkNotNullAndLog(String argName, Object value) {
    if (value == null || value.toString().length() == 0) {
      throw new IllegalArgumentException(argName + " is null or empty");
    }
    log.debug("{}: {}", argName, value);
  }

  protected static void checkNotNullAndLog(String argName, Object[] values) {
    if (values == null || values.length == 0) {
      throw new IllegalArgumentException(argName + " is null or zero-length");
    }
    for (Object value : values) {
      checkNotNullAndLog(argName, value);
    }
  }


  /**
   * <p>Looks up a {@link DataSource} by name from JNDI. "java:comp/env/" is prepended to the argument before looking up
   * the name in JNDI.</p>
   *
   * @param dataSourceName JNDI name where a {@link DataSource} is bound (e.g. "jdbc/taste")
   * @return {@link DataSource} under that JNDI name
   * @throws TasteException if a JNDI error occurs
   */
  protected static DataSource lookupDataSource(String dataSourceName) throws TasteException {
    Context context = null;
    try {
      context = new InitialContext();
      return (DataSource) context.lookup("java:comp/env/" + dataSourceName);
    } catch (NamingException ne) {
      throw new TasteException(ne);
    } finally {
      if (context != null) {
        try {
          context.close();
        } catch (NamingException ne) {
          log.warn("Error while closing Context; continuing...", ne);
        }
      }
    }
  }

  protected int getFetchSize() {
    return DEFAULT_FETCH_SIZE;
  }

  protected void advanceResultSet(ResultSet resultSet, int n) throws SQLException {
    resultSet.relative(n);
  }

}
