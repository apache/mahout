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

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Iterator;

/**
 * Iterable representing the lines of a text file. It can produce an {@link Iterator} over those lines. This assumes the
 * text file's lines are delimited in a manner consistent with how {@link java.io.BufferedReader} defines lines.
 *
 * This class will uncompress files that end in .zip or .gz accordingly, too.
 */
public final class FileLineIterable implements Iterable<String> {

  private final File file;
  private final Charset encoding;
  private final boolean skipFirstLine;

  /** Creates a {@link FileLineIterable} over a given file, assuming a UTF-8 encoding. */
  public FileLineIterable(File file) {
    this(file, Charset.forName("UTF-8"), false);
  }

  /** Creates a {@link FileLineIterable} over a given file, assuming a UTF-8 encoding. */
  public FileLineIterable(File file, boolean skipFirstLine) {
    this(file, Charset.forName("UTF-8"), skipFirstLine);
  }

  /** Creates a {@link FileLineIterable} over a given file, using the given encoding. */
  public FileLineIterable(File file, Charset encoding, boolean skipFirstLine) {
    this.file = file;
    this.encoding = encoding;
    this.skipFirstLine = skipFirstLine;
  }

  @Override
  public Iterator<String> iterator() {
    try {
      return new FileLineIterator(file, encoding, skipFirstLine);
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

}