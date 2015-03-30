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

package org.apache.mahout.cf.taste.impl.model;

import javax.sql.DataSource;

/**
 * <p>
 * An implementation for MySQL. The following statement would create a table suitable for use with this class:
 * </p>
 *
 * <p>
 *
 * <pre>
 * CREATE TABLE taste_id_migration (
 *   long_id BIGINT NOT NULL PRIMARY KEY,
 *   string_id VARCHAR(255) NOT NULL UNIQUE
 * )
 * </pre>
 *
 * </p>
 *
 * <p>
 * Separately, note that in a MySQL database, the following function calls will convert a string value into a
 * numeric value in the same way that the standard implementations in this package do. This may be useful in
 * writing SQL statements for use with
 * {@code AbstractJDBCDataModel} subclasses which convert string
 * column values to appropriate numeric values -- though this should be viewed as a temporary arrangement
 * since it will impact performance:
 * </p>
 *
 * <p>
 * {@code cast(conv(substring(md5([column name]), 1, 16),16,10) as signed)}
 * </p>
 */
public final class MySQLJDBCIDMigrator extends AbstractJDBCIDMigrator {
  
  public MySQLJDBCIDMigrator(DataSource dataSource) {
    this(dataSource, DEFAULT_MAPPING_TABLE,
        DEFAULT_LONG_ID_COLUMN, DEFAULT_STRING_ID_COLUMN);
  }
  
  public MySQLJDBCIDMigrator(DataSource dataSource,
                             String mappingTable,
                             String longIDColumn,
                             String stringIDColumn) {
    super(dataSource,
          "SELECT " + stringIDColumn + " FROM " + mappingTable + " WHERE " + longIDColumn + "=?",
          "INSERT IGNORE INTO " + mappingTable + " (" + longIDColumn + ',' + stringIDColumn + ") VALUES (?,?)");
  }
  
}
