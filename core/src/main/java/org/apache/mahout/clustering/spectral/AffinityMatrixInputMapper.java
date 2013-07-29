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

package org.apache.mahout.clustering.spectral;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Handles reading the files representing the affinity matrix. Since the affinity
 * matrix is representative of a graph, each line in all the files should
 * take the form:</p>
 *
 * {@code i,j,value}
 *
 * <p>where {@code i} and {@code j} are the {@code i}th and
 * {@code j} data points in the entire set, and {@code value}
 * represents some measurement of their relative absolute magnitudes. This
 * is, simply, a method for representing a graph textually.
 */
public class AffinityMatrixInputMapper
    extends Mapper<LongWritable, Text, IntWritable, DistributedRowMatrix.MatrixEntryWritable> {

  private static final Logger log = LoggerFactory.getLogger(AffinityMatrixInputMapper.class);

  private static final Pattern COMMA_PATTERN = Pattern.compile(",");

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

    String[] elements = COMMA_PATTERN.split(value.toString());
    log.debug("(DEBUG - MAP) Key[{}], Value[{}]", key.get(), value);

    // enforce well-formed textual representation of the graph
    if (elements.length != 3) {
      throw new IOException("Expected input of length 3, received "
                            + elements.length + ". Please make sure you adhere to "
                            + "the structure of (i,j,value) for representing a graph in text. "
                            + "Input line was: '" + value + "'.");
    }
    if (elements[0].isEmpty() || elements[1].isEmpty() || elements[2].isEmpty()) {
      throw new IOException("Found an element of 0 length. Please be sure you adhere to the structure of "
          + "(i,j,value) for  representing a graph in text.");
    }

    // parse the line of text into a DistributedRowMatrix entry,
    // making the row (elements[0]) the key to the Reducer, and
    // setting the column (elements[1]) in the entry itself
    DistributedRowMatrix.MatrixEntryWritable toAdd = new DistributedRowMatrix.MatrixEntryWritable();
    IntWritable row = new IntWritable(Integer.valueOf(elements[0]));
    toAdd.setRow(-1); // already set as the Reducer's key
    toAdd.setCol(Integer.valueOf(elements[1]));
    toAdd.setVal(Double.valueOf(elements[2]));
    context.write(row, toAdd);
  }
}
