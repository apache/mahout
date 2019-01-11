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
package org.apache.mahout.utils.io;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.Closeable;
import java.io.IOException;

/**
 * Writes data splitted in multiple Hadoop sequence files of approximate equal size. The data must consist
 * of key-value pairs, both of them of String type. All sequence files are created in the same
 * directory and named "chunk-0", "chunk-1", etc. 
 */
public final class ChunkedWriter implements Closeable {

  private final int maxChunkSizeInBytes;
  private final Path output;
  private SequenceFile.Writer writer;
  private int currentChunkID;
  private int currentChunkSize;
  private final FileSystem fs;
  private final Configuration conf;

  /** 
   * @param conf    needed by Hadoop to know what filesystem implementation to use.
   * @param chunkSizeInMB approximate size of each file, in Megabytes.
   * @param output        directory where the sequence files will be created.
   * @throws IOException
   */
  public ChunkedWriter(Configuration conf, int chunkSizeInMB, Path output) throws IOException {
    this.output = output;
    this.conf = conf;
    if (chunkSizeInMB > 1984) {
      chunkSizeInMB = 1984;
    }
    maxChunkSizeInBytes = chunkSizeInMB * 1024 * 1024;
    fs = FileSystem.get(output.toUri(), conf);
    currentChunkID = 0;
    writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID), Text.class, Text.class);
  }

  private Path getPath(int chunkID) {
    return new Path(output, "chunk-" + chunkID);
  }

  /** Writes a new key-value pair, creating a new sequence file if necessary.*/
  public void write(String key, String value) throws IOException {
    if (currentChunkSize > maxChunkSizeInBytes) {
      Closeables.close(writer, false);
      currentChunkID++;
      writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID), Text.class, Text.class);
      currentChunkSize = 0;
    }

    Text keyT = new Text(key);
    Text valueT = new Text(value);
    currentChunkSize += keyT.getBytes().length + valueT.getBytes().length; // Overhead
    writer.append(keyT, valueT);
  }

  @Override
  public void close() throws IOException {
    Closeables.close(writer, false);
  }
}

