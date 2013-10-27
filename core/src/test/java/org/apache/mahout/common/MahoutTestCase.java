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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.lang.reflect.Field;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.After;
import org.junit.Before;

public class MahoutTestCase extends org.apache.mahout.math.MahoutTestCase {

  /** "Close enough" value for floating-point comparisons. */
  public static final double EPSILON = 0.000001;

  private Path testTempDirPath;
  private FileSystem fs;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    testTempDirPath = null;
    fs = null;
  }

  @Override
  @After
  public void tearDown() throws Exception {
    if (testTempDirPath != null) {
      try {
        fs.delete(testTempDirPath, true);
      } catch (IOException e) {
        throw new IllegalStateException("Test file not found");
      }
      testTempDirPath = null;
      fs = null;
    }
    super.tearDown();
  }

  public final Configuration getConfiguration() throws IOException {
    Configuration conf = new Configuration();
    conf.set("hadoop.tmp.dir", getTestTempDir("hadoop" + Math.random()).getAbsolutePath());
    return conf;
  }

  protected final Path getTestTempDirPath() throws IOException {
    if (testTempDirPath == null) {
      fs = FileSystem.get(getConfiguration());
      long simpleRandomLong = (long) (Long.MAX_VALUE * Math.random());
      testTempDirPath = fs.makeQualified(
          new Path("/tmp/mahout-" + getClass().getSimpleName() + '-' + simpleRandomLong));
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
    if (dir && !fs.mkdirs(tempFileOrDir)) {
      throw new IOException("Could not create " + tempFileOrDir);
    }
    return tempFileOrDir;
  }

  /**
   * Try to directly set a (possibly private) field on an Object
   */
  protected static void setField(Object target, String fieldname, Object value)
    throws NoSuchFieldException, IllegalAccessException {
    Field field = findDeclaredField(target.getClass(), fieldname);
    field.setAccessible(true);
    field.set(target, value);
  }

  /**
   * Find a declared field in a class or one of it's super classes
   */
  private static Field findDeclaredField(Class<?> inClass, String fieldname) throws NoSuchFieldException {
    while (!Object.class.equals(inClass)) {
      for (Field field : inClass.getDeclaredFields()) {
        if (field.getName().equalsIgnoreCase(fieldname)) {
          return field;
        }
      }
      inClass = inClass.getSuperclass();
    }
    throw new NoSuchFieldException();
  }

  /**
   * @return a job option key string (--name) from the given option name
   */
  protected static String optKey(String optionName) {
    return AbstractJob.keyFor(optionName);
  }

  protected static void writeLines(File file, String... lines) throws IOException {
    Writer writer = new OutputStreamWriter(new FileOutputStream(file), Charsets.UTF_8);
    try {
      for (String line : lines) {
        writer.write(line);
        writer.write('\n');
      }
    } finally {
      Closeables.close(writer, false);
    }
  }
}
