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

package org.apache.mahout.cf.taste.impl.model.file;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.correlation.PearsonCorrelation;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * <p>Tests {@link FileDataModel}.</p>
 */
public final class FileDataModelTest extends TasteTestCase {

  private static final File testFile = new File("src/test/java/org/apache/mahout/cf/taste/impl/model/file/test1.txt");

  private DataModel model;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    model = new FileDataModel(testFile);
  }

  public void testFile() throws Exception {
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

  public void testGetItems() throws Exception {
    Iterable<? extends Item> items = model.getItems();
    assertNotNull(items);
    Iterator<? extends Item> it = items.iterator();
    assertNotNull(it);
    assertTrue(it.hasNext());
    assertEquals("123", it.next().getID());
    assertTrue(it.hasNext());
    assertEquals("234", it.next().getID());
    assertTrue(it.hasNext());
    assertEquals("456", it.next().getID());
    assertTrue(it.hasNext());
    assertEquals("654", it.next().getID());
    assertTrue(it.hasNext());
    assertEquals("789", it.next().getID());
    assertFalse(it.hasNext());
    try {
      it.next();
      fail("Should throw NoSuchElementException");
    } catch (NoSuchElementException nsee) {
      // good
    }
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

  public void testGetNumUsers() throws Exception {
    assertEquals(4, model.getNumUsers());
  }

  public void testNumUsersPreferring() throws Exception {
    assertEquals(2, model.getNumUsersWithPreferenceFor("456"));
    assertEquals(0, model.getNumUsersWithPreferenceFor("111"));
    assertEquals(0, model.getNumUsersWithPreferenceFor("111", "456"));
    assertEquals(2, model.getNumUsersWithPreferenceFor("123", "234"));
  }

  public void testSetPreference() throws Exception {
    try {
      model.setPreference(null, null, 0.0);
      fail("Should have thrown UnsupportedOperationException");
    } catch (UnsupportedOperationException uoe) {
      // good
    }
  }

  public void testRefresh() throws Exception {
    final AtomicBoolean initialized = new AtomicBoolean(false);
    Runnable initializer = new Runnable() {
      public void run() {
        try {
          model.getNumUsers();
          initialized.set(true);
        } catch (TasteException te) {
          // oops
        }
      }
    };
    new Thread(initializer).start();
    Thread.sleep(1000L); // wait a second for thread to start and call getNumUsers()
    model.getNumUsers(); // should block
    assertTrue(initialized.get());
    assertEquals(4, model.getNumUsers());
  }

  public void testToString() {
    assertTrue(model.toString().length() > 0);
  }

}
