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
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.nio.charset.Charset;

/**
 * <p>Tests {@link FileDataModel}.</p>
 */
public final class FileDataModelTest extends TasteTestCase {

  private static final String [] DATA = {
          "A123,456,0.1",
          "A123,789,0.6",
          "A123,654,0.7",
          "B234,123,0.5",
          "B234,234,1.0",
          "C345,789,0.6",
          "C345,654,0.7",
          "C345,123,1.0",
          "C345,234,0.5",
          "D456,456,0.1"};

  private DataModel model;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    File tmpLoc = new File(tmpDir, "fileDataModel");
    if (!tmpLoc.mkdirs()) {
      throw new IOException();
    }
    File testFile = File.createTempFile("test", ".txt", tmpLoc);
    testFile.deleteOnExit();
    PrintWriter writer =
        new PrintWriter(new OutputStreamWriter(new FileOutputStream(testFile), Charset.forName("UTF-8")));
    try {
      for (String data : DATA) {
        writer.println(data);
      }
    } finally {
      writer.close();
    }
    model = new FileDataModel(testFile);
  }

  public void testFile() throws Exception {
    UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, userSimilarity, model);
    Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);
    assertEquals(2, recommender.recommend("A123", 3).size());
    assertEquals(2, recommender.recommend("B234", 3).size());
    assertEquals(1, recommender.recommend("C345", 3).size());

    // Make sure this doesn't throw an exception
    model.refresh(null);
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
    model.setPreference("A123", "456", 0.2);
    assertEquals(0.2, model.getUser("A123").getPreferenceFor("456").getValue());
  }

  public void testRemovePreference() throws Exception {
    model.removePreference("A123", "456");
    assertNull(model.getUser("A123").getPreferenceFor("456"));
  }

  public void testRefresh() throws Exception {
    final AtomicBoolean initialized = new AtomicBoolean(false);
    Runnable initializer = new Runnable() {
      @Override
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
