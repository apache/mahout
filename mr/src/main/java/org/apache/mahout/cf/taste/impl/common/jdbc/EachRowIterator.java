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

import javax.sql.DataSource;
import java.io.Closeable;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Provides an {@link java.util.Iterator} over the result of an SQL query, as an iteration over the {@link ResultSet}.
 * While the same object will be returned from the iteration each time, it will be returned once for each row
 * of the result.
 */
final class EachRowIterator extends AbstractIterator<ResultSet> implements Closeable {

  private static final Logger log = LoggerFactory.getLogger(EachRowIterator.class);

  private final Connection connection;
  private final PreparedStatement statement;
  private final ResultSet resultSet;

  EachRowIterator(DataSource dataSource, String sqlQuery) throws SQLException {
    try {
      connection = dataSource.getConnection();
      statement = connection.prepareStatement(sqlQuery, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
      statement.setFetchDirection(ResultSet.FETCH_FORWARD);
      //statement.setFetchSize(getFetchSize());
      log.debug("Executing SQL query: {}", sqlQuery);
      resultSet = statement.executeQuery();
    } catch (SQLException sqle) {
      close();
      throw sqle;
    }
  }

  @Override
  protected ResultSet computeNext() {
    try {
      if (resultSet.next()) {
        return resultSet;
      } else {
        close();
        return null;
      }
    } catch (SQLException sqle) {
      close();
      throw new IllegalStateException(sqle);
    }
  }

  public void skip(int n) throws SQLException {
    try {
      resultSet.relative(n);
    } catch (SQLException sqle) {
      // Can't use relative on MySQL Connector/J; try advancing manually
      int i = 0;
      while (i < n && resultSet.next()) {
        i++;
      }
    }
  }

  @Override
  public void close() {
    IOUtils.quietClose(resultSet, statement, connection);
    endOfData();
  }

}
