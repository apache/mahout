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

import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.spectral.MatrixDiagonalizeJob.MatrixDiagonalizeMapper;
import org.apache.mahout.clustering.spectral.MatrixDiagonalizeJob.MatrixDiagonalizeReducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * <p>The MatrixDiagonalize task is pretty simple: given a matrix,
 * it sums the elements of the row, and sticks the sum in position (i, i) 
 * of a new matrix of identical dimensions to the original.</p>
 */
public class TestMatrixDiagonalizeJob extends MahoutTestCase {
  
  private static final double[][] RAW = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
  private static final int RAW_DIMENSIONS = 3;
  
  private static double rowSum(double [] row) {
    double sum = 0;
    for (double r : row) {
      sum += r;
    }
    return sum;
  }

  @Test
  public void testMatrixDiagonalizeMapper() throws Exception {
    MatrixDiagonalizeMapper mapper = new MatrixDiagonalizeMapper();
    Configuration conf = getConfiguration();
    conf.setInt(Keys.AFFINITY_DIMENSIONS, RAW_DIMENSIONS);
    
    // set up the dummy writers
    DummyRecordWriter<NullWritable, IntDoublePairWritable> writer =
      new DummyRecordWriter<NullWritable, IntDoublePairWritable>();
    Mapper<IntWritable, VectorWritable, NullWritable, IntDoublePairWritable>.Context 
      context = DummyRecordWriter.build(mapper, conf, writer);
    
    // perform the mapping
    for (int i = 0; i < RAW_DIMENSIONS; i++) {
      RandomAccessSparseVector toAdd = new RandomAccessSparseVector(RAW_DIMENSIONS);
      toAdd.assign(RAW[i]);
      mapper.map(new IntWritable(i), new VectorWritable(toAdd), context);
    }
    
    // check the number of the results
    assertEquals("Number of map results", RAW_DIMENSIONS,
        writer.getValue(NullWritable.get()).size());
  }
  
  @Test
 public void testMatrixDiagonalizeReducer() throws Exception {
    MatrixDiagonalizeMapper mapper = new MatrixDiagonalizeMapper();
    Configuration conf = getConfiguration();
    conf.setInt(Keys.AFFINITY_DIMENSIONS, RAW_DIMENSIONS);
    
    // set up the dummy writers
    DummyRecordWriter<NullWritable, IntDoublePairWritable> mapWriter = 
      new DummyRecordWriter<NullWritable, IntDoublePairWritable>();
    Mapper<IntWritable, VectorWritable, NullWritable, IntDoublePairWritable>.Context 
      mapContext = DummyRecordWriter.build(mapper, conf, mapWriter);
    
    // perform the mapping
    for (int i = 0; i < RAW_DIMENSIONS; i++) {
      RandomAccessSparseVector toAdd = new RandomAccessSparseVector(RAW_DIMENSIONS);
      toAdd.assign(RAW[i]);
      mapper.map(new IntWritable(i), new VectorWritable(toAdd), mapContext);
    }
    
    // now perform the reduction
    MatrixDiagonalizeReducer reducer = new MatrixDiagonalizeReducer();
    DummyRecordWriter<NullWritable, VectorWritable> redWriter = new
      DummyRecordWriter<NullWritable, VectorWritable>();
    Reducer<NullWritable, IntDoublePairWritable, NullWritable, VectorWritable>.Context
      redContext = DummyRecordWriter.build(reducer, conf, redWriter, 
      NullWritable.class, IntDoublePairWritable.class);
    
    // only need one reduction
    reducer.reduce(NullWritable.get(), mapWriter.getValue(NullWritable.get()), redContext);
    
    // first, make sure there's only one result
    List<VectorWritable> list = redWriter.getValue(NullWritable.get());
    assertEquals("Only a single resulting vector", 1, list.size());
    Vector v = list.get(0).get();
    for (int i = 0; i < v.size(); i++) {
      assertEquals("Element sum is correct", rowSum(RAW[i]), v.get(i),0.01);
    }
  }
}
