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

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.VectorIterable;

import java.io.IOException;
import java.io.Writer;

public class JWriterVectorWriter implements VectorWriter {
  protected Writer writer;

  public JWriterVectorWriter(Writer writer) {
    this.writer = writer;
  }

  @Override
  public long write(VectorIterable iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }

  @Override
  public long write(VectorIterable iterable, long maxDocs) throws IOException {
    long result = 0;

    for (Vector vector : iterable) {
      if (result >= maxDocs) {
        break;
      }
      writer.write(vector.asFormatString());
      writer.write("\n");

      result++;
    }
    return result;
  }

  @Override
  public void close() throws IOException {

  }
}
