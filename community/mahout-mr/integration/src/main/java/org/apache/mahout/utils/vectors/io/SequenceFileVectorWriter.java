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

import com.google.common.io.Closeables;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;


/**
 * Writes out Vectors to a SequenceFile.
 *
 * Closes the writer when done
 */
public class SequenceFileVectorWriter implements VectorWriter {
  private final SequenceFile.Writer writer;
  private long recNum = 0;
  public SequenceFileVectorWriter(SequenceFile.Writer writer) {
    this.writer = writer;
  }
  
  @Override
  public long write(Iterable<Vector> iterable, long maxDocs) throws IOException {

    for (Vector point : iterable) {
      if (recNum >= maxDocs) {
        break;
      }
      if (point != null) {
        writer.append(new LongWritable(recNum++), new VectorWritable(point));
      }
      
    }
    return recNum;
  }

  @Override
  public void write(Vector vector) throws IOException {
    writer.append(new LongWritable(recNum++), new VectorWritable(vector));

  }

  @Override
  public long write(Iterable<Vector> iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }
  
  @Override
  public void close() throws IOException {
    Closeables.close(writer, false);
  }
  
  public SequenceFile.Writer getWriter() {
    return writer;
  }
}
