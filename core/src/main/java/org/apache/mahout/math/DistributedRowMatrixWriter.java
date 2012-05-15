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
package org.apache.mahout.math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;

import java.io.IOException;

public final class DistributedRowMatrixWriter {

  private DistributedRowMatrixWriter() {
  }

  public static void write(Path outputDir, Configuration conf, Iterable<MatrixSlice> matrix) throws IOException {
    FileSystem fs = outputDir.getFileSystem(conf);
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outputDir,
        IntWritable.class, VectorWritable.class);
    IntWritable topic = new IntWritable();
    VectorWritable vector = new VectorWritable();
    for (MatrixSlice slice : matrix) {
      topic.set(slice.index());
      vector.set(slice.vector());
      writer.append(topic, vector);
    }
    writer.close();

  }

}
