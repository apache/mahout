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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.IOException;
import java.util.List;

public final class MatrixUtils {

  private MatrixUtils() {
  }

  public static void write(Path outputDir, Configuration conf, VectorIterable matrix)
    throws IOException {
    FileSystem fs = outputDir.getFileSystem(conf);
    fs.delete(outputDir, true);
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

  public static Matrix read(Configuration conf, Path... modelPaths) throws IOException {
    int numRows = -1;
    int numCols = -1;
    boolean sparse = false;
    List<Pair<Integer, Vector>> rows = Lists.newArrayList();
    for (Path modelPath : modelPaths) {
      for (Pair<IntWritable, VectorWritable> row
          : new SequenceFileIterable<IntWritable, VectorWritable>(modelPath, true, conf)) {
        rows.add(Pair.of(row.getFirst().get(), row.getSecond().get()));
        numRows = Math.max(numRows, row.getFirst().get());
        sparse = !row.getSecond().get().isDense();
        if (numCols < 0) {
          numCols = row.getSecond().get().size();
        }
      }
    }
    if (rows.isEmpty()) {
      throw new IOException(Arrays.toString(modelPaths) + " have no vectors in it");
    }
    numRows++;
    Vector[] arrayOfRows = new Vector[numRows];
    for (Pair<Integer, Vector> pair : rows) {
      arrayOfRows[pair.getFirst()] = pair.getSecond();
    }
    Matrix matrix;
    if (sparse) {
      matrix = new SparseRowMatrix(numRows, numCols, arrayOfRows);
    } else {
      matrix = new DenseMatrix(numRows, numCols);
      for (int i = 0; i < numRows; i++) {
        matrix.assignRow(i, arrayOfRows[i]);
      }
    }
    return matrix;
  }

  public static OpenObjectIntHashMap<String> readDictionary(Configuration conf, Path... dictPath) {
    OpenObjectIntHashMap<String> dictionary = new OpenObjectIntHashMap<String>();
    for (Path dictionaryFile : dictPath) {
      for (Pair<Writable, IntWritable> record
              : new SequenceFileIterable<Writable, IntWritable>(dictionaryFile, true, conf)) {
        dictionary.put(record.getFirst().toString(), record.getSecond().get());
      }
    }
    return dictionary;
  }

  public static String[] invertDictionary(OpenObjectIntHashMap<String> termIdMap) {
    int maxTermId = -1;
    for (String term : termIdMap.keys()) {
      maxTermId = Math.max(maxTermId, termIdMap.get(term));
    }
    maxTermId++;
    String[] dictionary = new String[maxTermId];
    for (String term : termIdMap.keys()) {
      dictionary[termIdMap.get(term)] = term;
    }
    return dictionary;
  }

}
