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

package org.apache.mahout.cf.taste.impl.common;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipInputStream;

/**
 * Iterates over the lines of a text file. This assumes the text file's lines are delimited in a manner consistent with
 * how {@link BufferedReader} defines lines.
 *
 * This class will uncompress files that end in .zip or .gz accordingly, too.
 */
public final class FileLineIterator implements SkippingIterator<String>, Closeable {

  private final BufferedReader reader;
  private String nextLine;

  /**
   * Creates a {@link FileLineIterator} over a given file, assuming a UTF-8 encoding.
   *
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException           if the file cannot be read
   */
  public FileLineIterator(File file) throws IOException {
    this(file, Charset.forName("UTF-8"), false);
  }

  /**
   * Creates a {@link FileLineIterator} over a given file, assuming a UTF-8 encoding.
   *
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException           if the file cannot be read
   */
  public FileLineIterator(File file, boolean skipFirstLine) throws IOException {
    this(file, Charset.forName("UTF-8"), skipFirstLine);
  }

  /**
   * Creates a {@link FileLineIterator} over a given file, using the given encoding.
   *
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException           if the file cannot be read
   */
  public FileLineIterator(File file, Charset encoding, boolean skipFirstLine) throws IOException {
    InputStream is = getFileInputStream(file);
    reader = new BufferedReader(new InputStreamReader(is, encoding));
    if (skipFirstLine) {
      reader.readLine();
    }
    nextLine = reader.readLine();
  }

  private static InputStream getFileInputStream(File file) throws IOException {
    InputStream is = new FileInputStream(file);
    String name = file.getName();
    if (name.endsWith(".gz")) {
      return new GZIPInputStream(is);
    } else if (name.endsWith(".zip")) {
      return new ZipInputStream(is);
    } else {
      return is;
    }
  }

  public String peek() {
    return nextLine;
  }

  @Override
  public boolean hasNext() {
    return nextLine != null;
  }

  @Override
  public String next() {
    if (nextLine == null) {
      throw new NoSuchElementException();
    }
    String result = nextLine;
    try {
      nextLine = reader.readLine();
    } catch (IOException ioe) {
      // Tough situation. Best to consider us done:
      close();
      throw new NoSuchElementException(ioe.toString());
    }
    if (nextLine == null) {
      close();
    }
    return result;
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void skip(int n) {
    try {
      for (int i = 0; i < n && nextLine != null; i++) {
        nextLine = reader.readLine();
      }
    } catch (IOException ioe) {
      close();
    }
  }

  @Override
  public void close() {
    nextLine = null;
    IOUtils.quietClose(reader);
  }

}
