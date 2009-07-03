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

package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;

import java.io.IOException;
import java.io.Writer;
import java.util.Iterator;


/**
 * Write ther TermInfo out to a {@link java.io.Writer}
 *
 **/
public class JWriterTermInfoWriter implements TermInfoWriter {

  protected Writer writer;
  protected String delimiter;
  protected String field;

  public JWriterTermInfoWriter(Writer writer, String delimiter, String field) {
    this.writer = writer;
    this.delimiter = delimiter;
    this.field = field;
  }

  @Override
  public void write(TermInfo ti) throws IOException {

    Iterator<TermEntry> entIter = ti.getAllEntries();

    writer.write(String.valueOf(ti.totalTerms(field)));
    writer.write("\n");
    writer.write("#term" + delimiter + "doc freq" + delimiter + "idx");
    writer.write("\n");
    while (entIter.hasNext()) {
      TermEntry entry = entIter.next();
      writer.write(entry.term);
      writer.write(delimiter);
      writer.write(String.valueOf(entry.docFreq));
      writer.write(delimiter);
      writer.write(String.valueOf(entry.termIdx));
      writer.write("\n");
    }
    writer.flush();
    writer.close();
  }

  /**
   * Does NOT close the underlying writer
   * @throws IOException
   */
  @Override
  public void close() throws IOException {

  }
}
