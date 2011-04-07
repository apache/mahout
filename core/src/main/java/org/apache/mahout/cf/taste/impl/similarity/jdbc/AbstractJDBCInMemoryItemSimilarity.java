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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;
import org.apache.mahout.cf.taste.impl.model.jdbc.ConnectionPoolDataSource;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.locks.ReentrantLock;

/**
 * loads all similarities from the database into RAM
 */
abstract class AbstractJDBCInMemoryItemSimilarity extends AbstractJDBCComponent implements ItemSimilarity {

  private ItemSimilarity delegate;

  private final DataSource dataSource;
  private final String getAllItemSimilaritiesSQL;
  private final ReentrantLock reloadLock;

  private static final Logger log = LoggerFactory.getLogger(AbstractJDBCInMemoryItemSimilarity.class);

  AbstractJDBCInMemoryItemSimilarity(DataSource dataSource, String getAllItemSimilaritiesSQL) {

    AbstractJDBCComponent.checkNotNullAndLog("getAllItemSimilaritiesSQL", getAllItemSimilaritiesSQL);

    if (!(dataSource instanceof ConnectionPoolDataSource)) {
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections " +
          "to the database itself, or database performance will be severely reduced.");
    }

    this.dataSource = dataSource;
    this.getAllItemSimilaritiesSQL = getAllItemSimilaritiesSQL;
    this.reloadLock = new ReentrantLock();

    reload();
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    return delegate.itemSimilarity(itemID1, itemID2);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    return delegate.itemSimilarities(itemID1, itemID2s);
  }

  @Override
  public long[] allSimilarItemIDs(long itemID) throws TasteException {
    return delegate.allSimilarItemIDs(itemID);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    log.debug("Reloading...");
    reload();
  }

  protected void reload() {
    if (reloadLock.tryLock()) {
      try {
        delegate = new GenericItemSimilarity(new JDBCSimilaritiesIterable());
      } finally {
        reloadLock.unlock();
      }
    }
  }

  class JDBCSimilaritiesIterable implements Iterable<GenericItemSimilarity.ItemItemSimilarity> {
    @Override
    public Iterator<GenericItemSimilarity.ItemItemSimilarity> iterator() {
      return new JDBCSimilaritiesIterator();
    }
  }

  private class JDBCSimilaritiesIterator implements Iterator<GenericItemSimilarity.ItemItemSimilarity> {

    private final Connection connection;
    private final PreparedStatement statement;
    private final ResultSet resultSet;
    private boolean closed;

    private JDBCSimilaritiesIterator() {
      try {
        connection = dataSource.getConnection();
        statement = connection.prepareStatement(getAllItemSimilaritiesSQL, ResultSet.TYPE_FORWARD_ONLY,
            ResultSet.CONCUR_READ_ONLY);
        statement.setFetchDirection(ResultSet.FETCH_FORWARD);
        statement.setFetchSize(getFetchSize());
        log.debug("Executing SQL query: {}", getAllItemSimilaritiesSQL);
        resultSet = statement.executeQuery();
        boolean anyResults = resultSet.next();
        if (!anyResults) {
          close();
        }
      } catch (SQLException e) {
        close();
        throw new IllegalStateException("Unable to read similarities!", e);
      }
    }

    @Override
    public boolean hasNext() {
      boolean nextExists = false;
      if (!closed) {
        try {
          if (resultSet.isAfterLast()) {
            close();
          } else {
            nextExists = true;
          }
        } catch (SQLException sqle) {
          log.warn("Unexpected exception while accessing ResultSet; continuing...", sqle);
          close();
        }
      }
      return nextExists;
    }

    @Override
    public GenericItemSimilarity.ItemItemSimilarity next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      try {
        GenericItemSimilarity.ItemItemSimilarity similarity = new GenericItemSimilarity.ItemItemSimilarity(
            resultSet.getLong(1), resultSet.getLong(2), resultSet.getDouble(3));
        resultSet.next();
        return similarity;
      } catch (SQLException e) {
        // No good way to handle this since we can't throw an exception
        log.warn("Exception while iterating", e);
        close();
        throw new IllegalStateException("Unable to read similarities!", e);
      }
    }

    /**
     * @throws UnsupportedOperationException
     */
    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

    private void close() {
      if (!closed) {
        closed = true;
        IOUtils.quietClose(resultSet, statement, connection);
      }
    }
  }

}
