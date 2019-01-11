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
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;

import javax.sql.DataSource;

public class SQL92JDBCInMemoryItemSimilarity extends AbstractJDBCInMemoryItemSimilarity {

  static final String DEFAULT_GET_ALL_ITEMSIMILARITIES_SQL =
      "SELECT " + AbstractJDBCItemSimilarity.DEFAULT_ITEM_A_ID_COLUMN + ", "
      + AbstractJDBCItemSimilarity.DEFAULT_ITEM_B_ID_COLUMN + ", "
      + AbstractJDBCItemSimilarity.DEFAULT_SIMILARITY_COLUMN + " FROM "
      + AbstractJDBCItemSimilarity.DEFAULT_SIMILARITY_TABLE;


  public SQL92JDBCInMemoryItemSimilarity() throws TasteException {
    this(AbstractJDBCComponent.lookupDataSource(AbstractJDBCComponent.DEFAULT_DATASOURCE_NAME),
         DEFAULT_GET_ALL_ITEMSIMILARITIES_SQL);
  }

  public SQL92JDBCInMemoryItemSimilarity(String dataSourceName) throws TasteException {
    this(AbstractJDBCComponent.lookupDataSource(dataSourceName), DEFAULT_GET_ALL_ITEMSIMILARITIES_SQL);
  }

  public SQL92JDBCInMemoryItemSimilarity(DataSource dataSource) {
    this(dataSource, DEFAULT_GET_ALL_ITEMSIMILARITIES_SQL);
  }

  public SQL92JDBCInMemoryItemSimilarity(DataSource dataSource, String getAllItemSimilaritiesSQL) {
    super(dataSource, getAllItemSimilaritiesSQL);
  }

}
