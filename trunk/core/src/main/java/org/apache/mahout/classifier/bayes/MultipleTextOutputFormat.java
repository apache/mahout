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

package org.apache.mahout.classifier.bayes;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Progressable;

/**
 * This class extends the MultipleOutputFormat, allowing to write the output
 * data to different output files in Text output format.
 */
public class MultipleTextOutputFormat<K, V> extends MultipleOutputFormat<K, V> {

  private TextOutputFormat<K, V> theTextOutputFormat;

  @Override
  protected RecordWriter<K, V> getBaseRecordWriter(FileSystem fs, Configuration conf, String name, Progressable arg3)
      throws IOException {
    if (theTextOutputFormat == null) {
      theTextOutputFormat = new TextOutputFormat<K, V>();
    }
    try {
      return theTextOutputFormat.getRecordWriter(new TaskAttemptContext(conf, new TaskAttemptID()));
    } catch (InterruptedException e) {
      // continue
    }
    return null;
  }

  @Override
  public RecordWriter<K, V> getRecordWriter(TaskAttemptContext job) throws IOException, InterruptedException {
    if (theTextOutputFormat == null) {
      theTextOutputFormat = new TextOutputFormat<K, V>();
    }
    return theTextOutputFormat.getRecordWriter(job);
  }
}
