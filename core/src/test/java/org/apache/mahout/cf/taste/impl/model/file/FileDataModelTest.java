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
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicBoolean;

/** <p>Tests {@link FileDataModel}.</p> */
public final class FileDataModelTest extends TasteTestCase {

  private static final String[] DATA = {
      "123,456,0.1",
      "123,789,0.6",
      "123,654,0.7",
      "234,123,0.5",
      "234,234,1.0",
      "345,789,0.6",
      "345,654,0.7",
      "345,123,1.0",
      "345,234,0.5",
      "456,456,0.1"};

  private DataModel model;
  private File testFile;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    File tmpLoc = new File(tmpDir, "fileDataModel");
    if (tmpLoc.exists()) {
      if (tmpLoc.isFile()) {
        throw new IOException("Temp directory is a file");
      }
    } else {
      if (!tmpLoc.mkdirs()) {
        throw new IOException("Could not create temp directory");
      }
    }
    testFile = File.createTempFile("test", ".txt", tmpLoc);
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
    assertEquals(2, recommender.recommend(123, 3).size());
    assertEquals(2, recommender.recommend(234, 3).size());
    assertEquals(1, recommender.recommend(345, 3).size());

    // Make sure this doesn't throw an exception
    model.refresh(null);
  }


  public void testTranspose() throws Exception {
    FileDataModel tModel = new FileDataModel(testFile, true);
    PreferenceArray userPrefs = tModel.getPreferencesFromUser(456);
    assertNotNull("user prefs are null and it shouldn't be", userPrefs);
    PreferenceArray pref = tModel.getPreferencesForItem(123);
    assertNotNull("pref is null and it shouldn't be", pref);
    assertEquals("pref Size: " + pref.length() + " is not: " + 3, 3, pref.length());
  }

  public void testGetItems() throws Exception {
    LongPrimitiveIterator it = model.getItemIDs();
    assertNotNull(it);
    assertTrue(it.hasNext());
    assertEquals(123, it.nextLong());
    assertTrue(it.hasNext());
    assertEquals(234, it.nextLong());
    assertTrue(it.hasNext());
    assertEquals(456, it.nextLong());
    assertTrue(it.hasNext());
    assertEquals(654, it.nextLong());
    assertTrue(it.hasNext());
    assertEquals(789, it.nextLong());
    assertFalse(it.hasNext());
    try {
      it.next();
      fail("Should throw NoSuchElementException");
    } catch (NoSuchElementException nsee) {
      // good
    }
  }

  public void testPreferencesForItem() throws Exception {
    PreferenceArray prefs = model.getPreferencesForItem(456);
    assertNotNull(prefs);
    Preference pref1 = prefs.get(0);
    assertEquals(123, pref1.getUserID());
    assertEquals(456, pref1.getItemID());
    Preference pref2 = prefs.get(1);
    assertEquals(456, pref2.getUserID());
    assertEquals(456, pref2.getItemID());
    assertEquals(2, prefs.length());
  }

  public void testGetNumUsers() throws Exception {
    assertEquals(4, model.getNumUsers());
  }

  public void testNumUsersPreferring() throws Exception {
    assertEquals(2, model.getNumUsersWithPreferenceFor(456));
    assertEquals(0, model.getNumUsersWithPreferenceFor(111));
    assertEquals(0, model.getNumUsersWithPreferenceFor(111, 456));
    assertEquals(2, model.getNumUsersWithPreferenceFor(123, 234));
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
