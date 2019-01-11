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

public class SQL92JDBCItemSimilarity extends AbstractJDBCItemSimilarity {

  public SQL92JDBCItemSimilarity() throws TasteException {
    this(DEFAULT_DATASOURCE_NAME);
  }

  public SQL92JDBCItemSimilarity(String dataSourceName) throws TasteException {
    this(lookupDataSource(dataSourceName));
  }

  public SQL92JDBCItemSimilarity(DataSource dataSource) {
    this(dataSource,
         DEFAULT_SIMILARITY_TABLE,
         DEFAULT_ITEM_A_ID_COLUMN,
         DEFAULT_ITEM_B_ID_COLUMN,
         DEFAULT_SIMILARITY_COLUMN);
  }

  public SQL92JDBCItemSimilarity(DataSource dataSource,
                                 String similarityTable,
                                 String itemAIDColumn,
                                 String itemBIDColumn,
                                 String similarityColumn) {
    super(dataSource,
          similarityTable,
          itemAIDColumn,
          itemBIDColumn, similarityColumn,
          "SELECT " + similarityColumn + " FROM " + similarityTable + " WHERE "
              + itemAIDColumn + "=? AND " + itemBIDColumn + "=?",
          "SELECT " + itemAIDColumn + ", " + itemBIDColumn + " FROM " + similarityTable + " WHERE "
              + itemAIDColumn + "=? OR " + itemBIDColumn + "=?");
  }

}
