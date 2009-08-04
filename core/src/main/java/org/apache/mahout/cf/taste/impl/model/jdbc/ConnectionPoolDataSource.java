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

import org.apache.commons.dbcp.ConnectionFactory;
import org.apache.commons.dbcp.PoolableConnectionFactory;
import org.apache.commons.dbcp.PoolingDataSource;
import org.apache.commons.pool.ObjectPool;
import org.apache.commons.pool.impl.StackObjectPool;

import javax.sql.DataSource;
import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;

/** <p>A wrapper {@link DataSource} which pools connections.</p> */
public final class ConnectionPoolDataSource implements DataSource {

  private final DataSource delegate;

  public ConnectionPoolDataSource(final DataSource underlyingDataSource) {
    if (underlyingDataSource == null) {
      throw new IllegalArgumentException("underlyingDataSource is null");
    }
    ConnectionFactory connectionFactory = new ConnectionFactory() {
      @Override
      public Connection createConnection() throws SQLException {
        Connection connection = underlyingDataSource.getConnection();
        connection.setTransactionIsolation(Connection.TRANSACTION_READ_UNCOMMITTED);
        connection.setHoldability(ResultSet.CLOSE_CURSORS_AT_COMMIT);
        return connection;
      }
    };
    ObjectPool objectPool = new StackObjectPool();
    objectPool.setFactory(new PoolableConnectionFactory(connectionFactory, objectPool, null, "SELECT 1", false, false));
    delegate = new PoolingDataSource(objectPool);
  }

  @Override
  public Connection getConnection() throws SQLException {
    return delegate.getConnection();
  }

  @Override
  public Connection getConnection(String username, String password) throws SQLException {
    return delegate.getConnection(username, password);
  }

  @Override
  public PrintWriter getLogWriter() throws SQLException {
    return delegate.getLogWriter();
  }

  @Override
  public void setLogWriter(PrintWriter printWriter) throws SQLException {
    delegate.setLogWriter(printWriter);
  }

  @Override
  public void setLoginTimeout(int timeout) throws SQLException {
    delegate.setLoginTimeout(timeout);
  }

  @Override
  public int getLoginTimeout() throws SQLException {
    return delegate.getLoginTimeout();
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException {
    return delegate.unwrap(iface);
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException {
    return delegate.isWrapperFor(iface);
  }

}
