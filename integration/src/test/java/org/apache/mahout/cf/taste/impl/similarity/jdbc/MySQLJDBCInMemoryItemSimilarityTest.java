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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.easymock.EasyMock;
import org.junit.Test;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

public class MySQLJDBCInMemoryItemSimilarityTest extends TasteTestCase {

  @Test
  public void testMemoryLoad() throws Exception {

    DataSource dataSource = EasyMock.createMock(DataSource.class);
    Connection connection = EasyMock.createMock(Connection.class);
    PreparedStatement statement = EasyMock.createMock(PreparedStatement.class);
    ResultSet resultSet = EasyMock.createMock(ResultSet.class);

    EasyMock.expect(dataSource.getConnection()).andReturn(connection);
    EasyMock.expect(connection.prepareStatement(MySQLJDBCInMemoryItemSimilarity.DEFAULT_GET_ALL_ITEMSIMILARITIES_SQL,
        ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY)).andReturn(statement);
    statement.setFetchDirection(ResultSet.FETCH_FORWARD);
    EasyMock.expect(statement.executeQuery()).andReturn(resultSet);

    EasyMock.expect(resultSet.next()).andReturn(true);

    EasyMock.expect(resultSet.getLong(1)).andReturn(1L);
    EasyMock.expect(resultSet.getLong(2)).andReturn(2L);
    EasyMock.expect(resultSet.getDouble(3)).andReturn(0.5);
    EasyMock.expect(resultSet.next()).andReturn(true);

    EasyMock.expect(resultSet.getLong(1)).andReturn(1L);
    EasyMock.expect(resultSet.getLong(2)).andReturn(3L);
    EasyMock.expect(resultSet.getDouble(3)).andReturn(0.4);
    EasyMock.expect(resultSet.next()).andReturn(true);

    EasyMock.expect(resultSet.getLong(1)).andReturn(3L);
    EasyMock.expect(resultSet.getLong(2)).andReturn(4L);
    EasyMock.expect(resultSet.getDouble(3)).andReturn(0.1);

    EasyMock.expect(resultSet.next()).andReturn(false);

    resultSet.close();
    statement.close();
    connection.close();

    EasyMock.replay(dataSource, connection, statement, resultSet);

    ItemSimilarity similarity = new MySQLJDBCInMemoryItemSimilarity(dataSource);

    assertEquals(0.5, similarity.itemSimilarity(1L, 2L), EPSILON);
    assertEquals(0.4, similarity.itemSimilarity(1L, 3L), EPSILON);
    assertEquals(0.1, similarity.itemSimilarity(3L, 4L), EPSILON);
    assertTrue(Double.isNaN(similarity.itemSimilarity(1L, 4L)));

    EasyMock.verify(dataSource, connection, statement, resultSet);
  }
}
