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

package org.apache.mahout.graph;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.common.MahoutTestCase;

import java.io.File;
import java.io.IOException;

public abstract class GraphTestCase extends MahoutTestCase {

  protected <T extends WritableComparable> void writeComponents(File destination, Configuration conf,
      Class<T> componentClass, T... components) throws IOException {
    Path path = new Path(destination.getAbsolutePath());
    FileSystem fs = FileSystem.get(path.toUri(), conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, componentClass, NullWritable.class);
    try {
      for (T component : components) {
        writer.append(component, NullWritable.get());
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
}
