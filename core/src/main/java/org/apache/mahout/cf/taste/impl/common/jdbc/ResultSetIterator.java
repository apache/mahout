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

import org.apache.mahout.common.iterator.TransformingIterator;

public abstract class ResultSetIterator<T> extends TransformingIterator<ResultSet,T> {

  protected ResultSetIterator(DataSource dataSource, String sqlQuery) throws SQLException {
    super(new EachRowIterator(dataSource, sqlQuery));
  }

  @Override
  protected final T transform(ResultSet in) {
    try {
      return parseElement(in);
    } catch (SQLException sqle) {
      throw new IllegalStateException(sqle);
    }
  }

  protected abstract T parseElement(ResultSet resultSet) throws SQLException;

  public void skip(int n) {
    if (n >= 1) {
      try {
        ((EachRowIterator) getDelegate()).skip(n);
      } catch (SQLException sqle) {
        throw new IllegalStateException(sqle);
      }
    }
  }

}
