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

package org.apache.mahout.common;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public abstract class MahoutTestCase extends org.apache.mahout.math.MahoutTestCase {

  private Path testTempDirPath;
  private FileSystem fs;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    testTempDirPath = null;
    fs = null;
  }

  @Override
  protected void tearDown() throws Exception {
    if (testTempDirPath != null) {
      fs.delete(testTempDirPath, true);
      testTempDirPath = null;
      fs = null;
    }
    super.tearDown();
  }

  protected final Path getTestTempDirPath() throws IOException {
    if (testTempDirPath == null) {
      fs = FileSystem.get(new Configuration());
      long simpleRandomLong = (long) (Long.MAX_VALUE * Math.random());
      testTempDirPath =
          fs.makeQualified(new Path("/tmp/mahout-" + getClass().getSimpleName() + '-' + simpleRandomLong));
      if (!fs.mkdirs(testTempDirPath)) {
        throw new IOException("Could not create " + testTempDirPath);
      }
      fs.deleteOnExit(testTempDirPath);
    }
    return testTempDirPath;
  }

  protected final Path getTestTempFilePath(String name) throws IOException {
    return getTestTempFileOrDirPath(name, false);
  }

  protected final Path getTestTempDirPath(String name) throws IOException {
    return getTestTempFileOrDirPath(name, true);
  }

  private Path getTestTempFileOrDirPath(String name, boolean dir) throws IOException {
    Path testTempDirPath = getTestTempDirPath();
    Path tempFileOrDir = fs.makeQualified(new Path(testTempDirPath, name));
    fs.deleteOnExit(tempFileOrDir);
    if (dir) {
      if (!fs.mkdirs(tempFileOrDir)) {
        throw new IOException("Could not create " + tempFileOrDir);
      }
    }
    return tempFileOrDir;
  }


}
