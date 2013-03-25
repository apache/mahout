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
import org.apache.mahout.cf.taste.impl.common.jdbc.ResultSetIterator;
import org.apache.mahout.cf.taste.impl.model.jdbc.ConnectionPoolDataSource;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Iterator;
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
      log.warn("You are not using ConnectionPoolDataSource. Make sure your DataSource pools connections "
               + "to the database itself, or database performance will be severely reduced.");
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
        delegate = new GenericItemSimilarity(new JDBCSimilaritiesIterable(dataSource, getAllItemSimilaritiesSQL));
      } finally {
        reloadLock.unlock();
      }
    }
  }

  private static final class JDBCSimilaritiesIterable implements Iterable<GenericItemSimilarity.ItemItemSimilarity> {

    private final DataSource dataSource;
    private final String getAllItemSimilaritiesSQL;

    private JDBCSimilaritiesIterable(DataSource dataSource, String getAllItemSimilaritiesSQL) {
      this.dataSource = dataSource;
      this.getAllItemSimilaritiesSQL = getAllItemSimilaritiesSQL;
    }

    @Override
    public Iterator<GenericItemSimilarity.ItemItemSimilarity> iterator() {
      try {
        return new JDBCSimilaritiesIterator(dataSource, getAllItemSimilaritiesSQL);
      } catch (SQLException sqle) {
        throw new IllegalStateException(sqle);
      }
    }
  }

  private static final class JDBCSimilaritiesIterator
      extends ResultSetIterator<GenericItemSimilarity.ItemItemSimilarity> {

    private JDBCSimilaritiesIterator(DataSource dataSource, String getAllItemSimilaritiesSQL) throws SQLException {
      super(dataSource, getAllItemSimilaritiesSQL);
    }

    @Override
    protected GenericItemSimilarity.ItemItemSimilarity parseElement(ResultSet resultSet) throws SQLException {
      return new GenericItemSimilarity.ItemItemSimilarity(resultSet.getLong(1),
                                                          resultSet.getLong(2),
                                                          resultSet.getDouble(3));
    }
  }

}
