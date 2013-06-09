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

import java.io.IOException;
import java.io.Writer;

import com.google.common.io.Closeables;
import org.apache.mahout.math.Vector;

/**
 * Write out the vectors to any {@link Writer} using {@link Vector#asFormatString()},
 * one per line by default.
 */
public class TextualVectorWriter implements VectorWriter {

  private final Writer writer;
  
  public TextualVectorWriter(Writer writer) {
    this.writer = writer;
  }

  protected Writer getWriter() {
    return writer;
  }
  
  @Override
  public long write(Iterable<Vector> iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }
  
  @Override
  public long write(Iterable<Vector> iterable, long maxDocs) throws IOException {
    long result = 0;
    for (Vector vector : iterable) {
      if (result >= maxDocs) {
        break;
      }
      write(vector);
      result++;
    }
    return result;
  }

  @Override
  public void write(Vector vector) throws IOException {
    writer.write(vector.asFormatString());
    writer.write('\n');
  }

  @Override
  public void close() throws IOException {
    Closeables.close(writer, false);
  }
}
