/*
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

import java.io.File;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.IDMigrator;
import org.junit.Before;
import org.junit.Test;

/**
 * Tests {@link FileIDMigrator}
 */
public final class FileIDMigratorTest extends TasteTestCase {

  private static final String[] STRING_IDS = {
      "dog",
      "cow" };

  private static final String[] UPDATED_STRING_IDS = {
      "dog",
      "cow",
      "donkey" };

  private File testFile;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    testFile = getTestTempFile("test.txt");
    writeLines(testFile, STRING_IDS);
  }

  @Test
  public void testLoadFromFile() throws Exception {
    IDMigrator migrator = new FileIDMigrator(testFile);
    long dogAsLong = migrator.toLongID("dog");
    long cowAsLong = migrator.toLongID("cow");
    long donkeyAsLong = migrator.toLongID("donkey");
    assertEquals("dog", migrator.toStringID(dogAsLong));
    assertEquals("cow", migrator.toStringID(cowAsLong));
    assertNull(migrator.toStringID(donkeyAsLong));
  }

  @Test
  public void testNoRefreshAfterFileUpdate() throws Exception {
    IDMigrator migrator = new FileIDMigrator(testFile, 0L);

    /* call a method to make sure the original file is loaded */
    long dogAsLong = migrator.toLongID("dog");
    migrator.toStringID(dogAsLong);

    /* change the underlying file,
     * we have to wait at least a second to see the change in the file's lastModified timestamp */
    Thread.sleep(2000L);
    writeLines(testFile, UPDATED_STRING_IDS);

    /* we shouldn't see any changes in the data as we have not yet refreshed */
    long cowAsLong = migrator.toLongID("cow");
    long donkeyAsLong = migrator.toLongID("donkey");
    assertEquals("dog", migrator.toStringID(dogAsLong));
    assertEquals("cow", migrator.toStringID(cowAsLong));
    assertNull(migrator.toStringID(donkeyAsLong));
  }

  @Test
  public void testRefreshAfterFileUpdate() throws Exception {
    IDMigrator migrator = new FileIDMigrator(testFile, 0L);

    /* call a method to make sure the original file is loaded */
    long dogAsLong = migrator.toLongID("dog");
    migrator.toStringID(dogAsLong);

    /* change the underlying file,
     * we have to wait at least a second to see the change in the file's lastModified timestamp */
    Thread.sleep(2000L);
    writeLines(testFile, UPDATED_STRING_IDS);

    migrator.refresh(null);

    long cowAsLong = migrator.toLongID("cow");
    long donkeyAsLong = migrator.toLongID("donkey");
    assertEquals("dog", migrator.toStringID(dogAsLong));
    assertEquals("cow", migrator.toStringID(cowAsLong));
    assertEquals("donkey", migrator.toStringID(donkeyAsLong));
  }
}
