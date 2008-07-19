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

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.io.File;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.io.BufferedReader;
import java.io.Reader;
import java.io.IOException;
import java.io.Closeable;

/**
 * Iterates over the lines of a text file. This assumes the text file is UTF-8 encoded
 * and that its lines are delimited in a manner consistent with how {@link BufferedReader}
 * defines lines.
 */
public final class FileLineIterator implements Iterator<String>, Closeable {

  private final BufferedReader reader;
  private String nextLine;

  /**
   * @throws FileNotFoundException if the file does not exist
   * @throws IOException if the file cannot be read
   */
  public FileLineIterator(File file) throws IOException {
    InputStream is = new FileInputStream(file);
    Reader fileReader;
    try {
      fileReader = new InputStreamReader(is, "UTF8");
    } catch (UnsupportedEncodingException uee) {
      throw new AssertionError(uee);
    }
    reader = new BufferedReader(fileReader);
    nextLine = reader.readLine();
  }

  public boolean hasNext() {
    return nextLine != null;
  }

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
  public void remove() {
    throw new UnsupportedOperationException();
  }

  public void close() {
    nextLine = null;
    IOUtils.quietClose(reader);
  }

}
