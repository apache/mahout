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

package org.apache.mahout.cf.taste.impl.model.jdbc;

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.correlation.PearsonCorrelation;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import static org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel.DEFAULT_ITEM_ID_COLUMN;
import static org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel.DEFAULT_PREFERENCE_COLUMN;
import static org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel.DEFAULT_PREFERENCE_TABLE;
import static org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel.DEFAULT_USER_ID_COLUMN;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.MemoryDiffStorage;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.util.Collections;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * <p>Tests {@link MySQLJDBCDataModel}.</p>
 *
 * <p>Requires a MySQL 4.x+ database running on the localhost, with a passwordless user named "mysql" available,
 * a database named "test".</p>
 */
public final class MySQLJDBCDataModelTest extends TasteTestCase {

  private DataModel model;

  @Override
  public void setUp() throws Exception {

    super.setUp();

    MysqlDataSource dataSource = new MysqlDataSource();
    dataSource.setUser("mysql");
    dataSource.setDatabaseName("test");
    dataSource.setServerName("localhost");

    Connection connection = dataSource.getConnection();
    try {

      PreparedStatement dropStatement =
        connection.prepareStatement("DROP TABLE IF EXISTS " + DEFAULT_PREFERENCE_TABLE);
      try {
        dropStatement.execute();
      } finally {
        dropStatement.close();
      }

      PreparedStatement createStatement =
        connection.prepareStatement("CREATE TABLE " + DEFAULT_PREFERENCE_TABLE + " (" +
                                    DEFAULT_USER_ID_COLUMN + " VARCHAR(4) NOT NULL, " +
                                    DEFAULT_ITEM_ID_COLUMN + " VARCHAR(4) NOT NULL, " +
                                    DEFAULT_PREFERENCE_COLUMN + " FLOAT NOT NULL, " +
                                    "PRIMARY KEY (" + DEFAULT_USER_ID_COLUMN + ", " +
                                    DEFAULT_ITEM_ID_COLUMN + "), " +
                                    "INDEX (" + DEFAULT_USER_ID_COLUMN + "), " +
                                    "INDEX (" + DEFAULT_ITEM_ID_COLUMN + ") )");
      try {
        createStatement.execute();
      } finally {
        createStatement.close();
      }

      PreparedStatement insertStatement =
        connection.prepareStatement("INSERT INTO " + DEFAULT_PREFERENCE_TABLE + " VALUES (?, ?, ?)");
      try {
        String[] users =
                new String[]{"A123", "A123", "A123", "B234", "B234", "C345", "C345", "C345", "C345", "D456"};
        String[] itemIDs =
                new String[]{"456", "789", "654", "123", "234", "789", "654", "123", "234", "456"};
        double[] preferences = new double[]{0.1, 0.6, 0.7, 0.5, 1.0, 0.6, 0.7, 1.0, 0.5, 0.1};
        for (int i = 0; i < users.length; i++) {
          insertStatement.setString(1, users[i]);
          insertStatement.setString(2, itemIDs[i]);
          insertStatement.setDouble(3, preferences[i]);
          insertStatement.execute();
        }
      } finally {
        insertStatement.close();
      }

    } finally {
      connection.close();
    }

    model = new MySQLJDBCDataModel(dataSource);
  }

  public void testStatements() throws Exception {
    assertEquals(4, model.getNumUsers());
    assertEquals(5, model.getNumItems());
    assertEquals(new GenericUser<String>("A123", Collections.<Preference>emptyList()), model.getUser("A123"));
    assertEquals(new GenericItem<String>("456"), model.getItem("456"));
    Preference pref = model.getUser("A123").getPreferenceFor("456");
    assertNotNull(pref);
    assertEquals(0.1, pref.getValue(), EPSILON);
    model.setPreference("A123", "456", 0.2);
    Preference pref1 = model.getUser("A123").getPreferenceFor("456");
    assertNotNull(pref1);
    assertEquals(0.2, pref1.getValue(), EPSILON);
    model.removePreference("A123", "456");
    assertNull(model.getUser("A123").getPreferenceFor("456"));
  }

  public void testDatabase() throws Exception {
    UserCorrelation userCorrelation = new PearsonCorrelation(model);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, userCorrelation, model);
    Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userCorrelation);
    assertEquals(2, recommender.recommend("A123", 3).size());
    assertEquals(2, recommender.recommend("B234", 3).size());
    assertEquals(1, recommender.recommend("C345", 3).size());

    // Make sure this doesn't throw an exception
    model.refresh();
  }

  public void testItem() throws Exception {
    assertEquals("456", model.getItem("456").getID());
  }

  public void testPreferencesForItem() throws Exception {
    Iterable<? extends Preference> prefs = model.getPreferencesForItem("456");
    assertNotNull(prefs);
    Iterator<? extends Preference> it = prefs.iterator();
    assertNotNull(it);
    assertTrue(it.hasNext());
    Preference pref1 = it.next();
    assertEquals("A123", pref1.getUser().getID());
    assertEquals("456", pref1.getItem().getID());
    assertTrue(it.hasNext());
    Preference pref2 = it.next();
    assertEquals("D456", pref2.getUser().getID());
    assertEquals("456", pref2.getItem().getID());
    assertFalse(it.hasNext());
    try {
      it.next();
      fail("Should throw NoSuchElementException");
    } catch (NoSuchElementException nsee) {
      // good
    }
  }

  public void testPreferencesForItemOrder() throws Exception {
    for (Item item : model.getItems()) {
      Iterable<? extends Preference> prefs = model.getPreferencesForItem(item.getID());
      User lastUser = null;
      for (Preference pref : prefs) {
        User thisUser = pref.getUser();
        if (lastUser != null) {
          String lastID = (String) lastUser.getID();
          String ID = (String) thisUser.getID();
          assertTrue(lastID.compareTo(ID) < 0);
        }
        lastUser = thisUser;
      }
    }
  }

  public void testSetPreference() throws Exception {
    model.setPreference("A123", "409", 2.0);
    Preference pref = model.getUser("A123").getPreferenceFor("409");
    assertNotNull(pref);
    assertEquals(2.0, pref.getValue());
    model.setPreference("A123", "409", 1.0);
    Preference pref1 = model.getUser("A123").getPreferenceFor("409");
    assertNotNull(pref1);
    assertEquals(1.0, pref1.getValue());
  }

  public void testSetPrefMemoryDiffUpdates() throws Exception {
    DiffStorage diffStorage = new MemoryDiffStorage(model, false, false, Long.MAX_VALUE);
    Recommender recommender = new SlopeOneRecommender(model, true, true, diffStorage);
    assertEquals(0.5, diffStorage.getDiff("456", "789").getAverage(), EPSILON);
    recommender.setPreference("A123", "456", 0.7);
    assertEquals(-0.1, diffStorage.getDiff("456", "789").getAverage(), EPSILON);
  }

}
