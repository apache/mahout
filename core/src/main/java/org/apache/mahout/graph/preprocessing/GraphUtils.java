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

package org.apache.mahout.graph.preprocessing;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/** utility methods for working with graphs */
public class GraphUtils {

  private GraphUtils() {}

  //TODO do this in parallel?
  public static int indexVertices(Configuration conf, Path verticesPath, Path indexPath) throws IOException {
    FileSystem fs = FileSystem.get(verticesPath.toUri(), conf);
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, indexPath, IntWritable.class, Vertex.class);
    int index = 0;

    try {
      for (FileStatus fileStatus : fs.listStatus(verticesPath)) {
        InputStream in = HadoopUtil.openStream(fileStatus.getPath(), conf);
        try {
          for (String line : new FileLineIterable(in)) {
            writer.append(new IntWritable(index++), new Vertex(Long.parseLong(line)));
          }
        } finally {
          Closeables.closeQuietly(in);
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }

    return index;
  }

  public static List<Integer> readVerticesIndexes(Configuration conf, Path path) throws IOException {
    List<Integer> indexes = Lists.newArrayList();
    InputStream in = HadoopUtil.openStream(path, conf);
    try {
      for (String line : new FileLineIterable(in)) {
        indexes.add(Integer.parseInt(line));
      }
    } finally {
      Closeables.closeQuietly(in);
    }

    return indexes;
  }

  public static void persistVector(Configuration conf, Path path, Vector vector) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    DataOutputStream out = fs.create(path, true);
    try {
      VectorWritable.writeVector(out, vector);
    } finally {
      Closeables.closeQuietly(out);
    }
  }
}