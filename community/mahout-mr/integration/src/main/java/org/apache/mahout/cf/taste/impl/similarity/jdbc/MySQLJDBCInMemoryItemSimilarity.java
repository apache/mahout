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

import org.apache.mahout.cf.taste.common.TasteException;

import javax.sql.DataSource;

public class MySQLJDBCInMemoryItemSimilarity extends SQL92JDBCInMemoryItemSimilarity {

  public MySQLJDBCInMemoryItemSimilarity() throws TasteException {
  }

  public MySQLJDBCInMemoryItemSimilarity(String dataSourceName) throws TasteException {
    super(dataSourceName);
  }

  public MySQLJDBCInMemoryItemSimilarity(DataSource dataSource) {
    super(dataSource);
  }

  public MySQLJDBCInMemoryItemSimilarity(DataSource dataSource, String getAllItemSimilaritiesSQL) {
    super(dataSource, getAllItemSimilaritiesSQL);
  }

  @Override
  protected int getFetchSize() {
    // Need to return this for MySQL Connector/J to make it use streaming mode
    return Integer.MIN_VALUE;
  }

}
