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
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Iterator;

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;

public abstract class ResultSetIterator<T> extends ForwardingIterator<T> {

  private final Iterator<T> delegate;
  private final EachRowIterator rowDelegate;

  protected ResultSetIterator(DataSource dataSource, String sqlQuery) throws SQLException {
    this.rowDelegate = new EachRowIterator(dataSource, sqlQuery);
    delegate = Iterators.transform(rowDelegate,
      new Function<ResultSet, T>() {
        @Override
        public T apply(ResultSet from) {
          try {
            return parseElement(from);
          } catch (SQLException sqle) {
            throw new IllegalStateException(sqle);
          }
        }
      });
  }

  @Override
  protected Iterator<T> delegate() {
    return delegate;
  }

  protected abstract T parseElement(ResultSet resultSet) throws SQLException;

  public void skip(int n) {
    if (n >= 1) {
      try {
        rowDelegate.skip(n);
      } catch (SQLException sqle) {
        throw new IllegalStateException(sqle);
      }
    }
  }

}
