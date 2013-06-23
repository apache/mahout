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

import java.io.IOException;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

/**
 * RecordReader used with the MultipleTextFileInputFormat class to read full files as
 * k/v pairs and groups of files as single input splits.
 */
public class WholeFileRecordReader extends RecordReader<IntWritable, BytesWritable> {

  private FileSplit fileSplit;
  private boolean processed = false;
  private Configuration configuration;
  private BytesWritable value = new BytesWritable();
  private IntWritable index;

  public WholeFileRecordReader(CombineFileSplit fileSplit, TaskAttemptContext taskAttemptContext, Integer idx)
    throws IOException {
    this.fileSplit = new FileSplit(fileSplit.getPath(idx), fileSplit.getOffset(idx),
       fileSplit.getLength(idx), fileSplit.getLocations());
    this.configuration = taskAttemptContext.getConfiguration();
    this.index = new IntWritable(idx);
  }

  @Override
  public IntWritable getCurrentKey() {
    return index;
  }

  @Override
  public BytesWritable getCurrentValue() {
    return value;
  }

  @Override
  public float getProgress() throws IOException {
    return processed ? 1.0f : 0.0f;
  }

  @Override
  public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext)
    throws IOException, InterruptedException {

  }

  @Override
  public boolean nextKeyValue() throws IOException {
    if (!processed) {
      byte[] contents = new byte[(int) fileSplit.getLength()];
      Path file = fileSplit.getPath();
      FileSystem fs = file.getFileSystem(this.configuration);
      FSDataInputStream in = null;
      try {
        if (!fs.isFile(file)) {
          return false;
        }
        in = fs.open(file);
        IOUtils.readFully(in, contents, 0, contents.length);
        value.setCapacity(contents.length);
        value.set(contents, 0, contents.length);
      } finally {
        Closeables.close(in, false);
      }
      processed = true;
      return true;
    }
    return false;
  }

  @Override
  public void close() throws IOException {
  }
}