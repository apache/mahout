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

import org.apache.mahout.utils.vectors.VectorIterable;
import org.apache.mahout.matrix.Vector;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;

import java.io.IOException;


/**
 * Closes the writer when done
 *
 **/
public class SequenceFileVectorWriter implements VectorWriter {
  protected SequenceFile.Writer writer;

  public SequenceFileVectorWriter(SequenceFile.Writer writer) {
    this.writer = writer;
  }

  @Override
  public long write(VectorIterable iterable, long maxDocs) throws IOException {
    long recNum = 0;
    for (Vector point : iterable) {
      if (recNum >= maxDocs) {
        break;
      }
      //point.write(dataOut);
      writer.append(new LongWritable(recNum++), point);

    }
    return recNum;
  }

  @Override
  public long write(VectorIterable iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }

  @Override
  public void close() throws IOException {
    if (writer != null) {
      writer.close();
    }
  }

  public SequenceFile.Writer getWriter() {
    return writer;
  }
}
