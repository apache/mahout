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
package org.apache.mahout.text;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.Closeable;
import java.io.IOException;

public final class ChunkedWriter implements Closeable {

  private final int maxChunkSizeInBytes;
  private final Path output;
  private SequenceFile.Writer writer;
  private int currentChunkID;
  private int currentChunkSize;
  private final FileSystem fs;
  private final Configuration conf;

  public ChunkedWriter(Configuration conf, int chunkSizeInMB, Path output) throws IOException {
    this.output = output;
    this.conf = conf;
    if (chunkSizeInMB > 1984) {
      chunkSizeInMB = 1984;
    }
    maxChunkSizeInBytes = chunkSizeInMB * 1024 * 1024;
    fs = FileSystem.get(conf);
    currentChunkID = 0;
    writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID), Text.class, Text.class);
  }

  private Path getPath(int chunkID) {
    return new Path(output, "chunk-" + chunkID);
  }

  public void write(String key, String value) throws IOException {
    if (currentChunkSize > maxChunkSizeInBytes) {
      Closeables.closeQuietly(writer);
      writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID++), Text.class, Text.class);
      currentChunkSize = 0;
    }

    Text keyT = new Text(key);
    Text valueT = new Text(value);
    currentChunkSize += keyT.getBytes().length + valueT.getBytes().length; // Overhead
    writer.append(keyT, valueT);
  }

  @Override
  public void close() throws IOException {
    Closeables.closeQuietly(writer);
  }
}

