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

import javax.sql.DataSource;

import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>
 * An {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity} backed by a MySQL database
 * and accessed via JDBC. It may work with other JDBC
 * databases. By default, this class assumes that there is a {@link DataSource} available under the JNDI name
 * "jdbc/taste", which gives access to a database with a "taste_item_similarity" table with the following
 * schema:
 * </p>
 * 
 * <table>
 * <tr>
 * <th>item_id_a</th>
 * <th>item_id_b</th>
 * <th>similarity</th>
 * </tr>
 * <tr>
 * <td>ABC</td>
 * <td>DEF</td>
 * <td>0.9</td>
 * </tr>
 * <tr>
 * <td>DEF</td>
 * <td>EFG</td>
 * <td>0.1</td>
 * </tr>
 * </table>
 * 
 * <p>
 * For example, the following command sets up a suitable table in MySQL, complete with primary key and
 * indexes:
 * </p>
 * 
 * <p>
 * 
 * <pre>
 * CREATE TABLE taste_item_similarity (
 *   item_id_a BIGINT NOT NULL,
 *   item_id_b BIGINT NOT NULL,
 *   similarity FLOAT NOT NULL,
 *   PRIMARY KEY (item_id_a, item_id_b),
 * )
 * </pre>
 * 
 * </p>
 * 
 * <p>
 * Note that for each row, item_id_a should be less than item_id_b. It is redundant to store it both ways,
 * so the pair is always stored as a pair with the lesser one first.
 * 
 * @see org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel
 */
public class MySQLJDBCItemSimilarity extends SQL92JDBCItemSimilarity {

  public MySQLJDBCItemSimilarity() throws TasteException {
  }

  public MySQLJDBCItemSimilarity(String dataSourceName) throws TasteException {
    super(dataSourceName);
  }

  public MySQLJDBCItemSimilarity(DataSource dataSource) {
    super(dataSource);
  }

  public MySQLJDBCItemSimilarity(DataSource dataSource,
                                 String similarityTable,
                                 String itemAIDColumn,
                                 String itemBIDColumn,
                                 String similarityColumn) {
    super(dataSource, similarityTable, itemAIDColumn, itemBIDColumn, similarityColumn);
  }
  
  @Override
  protected int getFetchSize() {
    // Need to return this for MySQL Connector/J to make it use streaming mode
    return Integer.MIN_VALUE;
  }
  
}

