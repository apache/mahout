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

package org.apache.mahout.math;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;

import org.apache.mahout.common.RandomUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;

/**
 * Superclass of all Mahout test cases.
 */
public abstract class MahoutTestCase extends Assert {

  /** "Close enough" value for floating-point comparisons. */
  public static final double EPSILON = 0.000001;
  
  private File testTempDir;

  @Before
  public void setUp() throws Exception {
    testTempDir = null;
    RandomUtils.useTestSeed();
  }

  @After
  public void tearDown() throws Exception {
    if (testTempDir != null) {
      new DeletingVisitor().accept(testTempDir);
    }
  }

  protected final File getTestTempDir() throws IOException {
    if (testTempDir == null) {
      String systemTmpDir = System.getProperty("java.io.tmpdir");
      long simpleRandomLong = (long) (Long.MAX_VALUE * Math.random());
      testTempDir = new File(systemTmpDir, "mahout-" + getClass().getSimpleName() + '-' + simpleRandomLong);
      if (!testTempDir.mkdir()) {
        throw new IOException("Could not create " + testTempDir);
      }
      testTempDir.deleteOnExit();
    }
    return testTempDir;
  }

  protected final File getTestTempFile(String name) throws IOException {
    return getTestTempFileOrDir(name, false);
  }

  protected final File getTestTempDir(String name) throws IOException {
    return getTestTempFileOrDir(name, true);
  }

  private File getTestTempFileOrDir(String name, boolean dir) throws IOException {
    File f = new File(getTestTempDir(), name);
    f.deleteOnExit();
    if (dir && !f.mkdirs()) {
      throw new IOException("Could not make directory " + f);
    }
    return f;
  }

  private static class DeletingVisitor implements FileFilter {
    @Override
    public boolean accept(File f) {
      if (!f.isFile()) {
        f.listFiles(this);
      }
      f.delete();
      return false;
    }
  }

}
