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

import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.logging.Logger;

import javax.sql.DataSource;

import org.apache.commons.dbcp.ConnectionFactory;
import org.apache.commons.dbcp.PoolableConnectionFactory;
import org.apache.commons.dbcp.PoolingDataSource;
import org.apache.commons.pool.impl.GenericObjectPool;

import com.google.common.base.Preconditions;

/**
 * <p>
 * A wrapper {@link DataSource} which pools connections.
 * </p>
 */
public final class ConnectionPoolDataSource implements DataSource {
  
  private final DataSource delegate;
  
  public ConnectionPoolDataSource(DataSource underlyingDataSource) {
    Preconditions.checkNotNull(underlyingDataSource);
    ConnectionFactory connectionFactory = new ConfiguringConnectionFactory(underlyingDataSource);
    GenericObjectPool objectPool = new GenericObjectPool();
    objectPool.setTestOnBorrow(false);
    objectPool.setTestOnReturn(false);
    objectPool.setTestWhileIdle(true);
    objectPool.setTimeBetweenEvictionRunsMillis(60 * 1000L);
    // Constructor actually sets itself as factory on pool
    new PoolableConnectionFactory(connectionFactory, objectPool, null, "SELECT 1", false, false);
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

  // This exists for compatibility with Java 7 / JDBC 4.1, but doesn't exist
  // in Java 6. In Java 7 it would @Override, but not in 6.
  // @Override
  public Logger getParentLogger() throws SQLFeatureNotSupportedException {
    throw new SQLFeatureNotSupportedException();
  }
  
  private static class ConfiguringConnectionFactory implements ConnectionFactory {
    
    private final DataSource underlyingDataSource;
    
    ConfiguringConnectionFactory(DataSource underlyingDataSource) {
      this.underlyingDataSource = underlyingDataSource;
    }
    
    @Override
    public Connection createConnection() throws SQLException {
      Connection connection = underlyingDataSource.getConnection();
      connection.setTransactionIsolation(Connection.TRANSACTION_READ_UNCOMMITTED);
      connection.setHoldability(ResultSet.CLOSE_CURSORS_AT_COMMIT);
      return connection;
    }
  }
  
}
