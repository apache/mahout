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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.spectral.VectorMatrixMultiplicationJob.VectorMatrixMultiplicationMapper;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * <p>This test ensures that a Vector can be successfully multiplied
 * with a matrix.</p>
 */
public class TestVectorMatrixMultiplicationJob extends MahoutTestCase {
  
  private static final double [][] MATRIX = { {1, 1}, {2, 3} };
  private static final double [] VECTOR = {9, 16};
  
  @Test
  public void testVectorMatrixMultiplicationMapper() throws Exception {
    VectorMatrixMultiplicationMapper mapper = new VectorMatrixMultiplicationMapper();
    Configuration conf = getConfiguration();
    
    // set up all the parameters for the job
    Vector toSave = new DenseVector(VECTOR);
    DummyRecordWriter<IntWritable, VectorWritable> writer = new 
      DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context
      context = DummyRecordWriter.build(mapper, conf, writer);
    mapper.setup(toSave);
    
    // run the job
    for (int i = 0; i < MATRIX.length; i++) {
      Vector v = new RandomAccessSparseVector(MATRIX[i].length);
      v.assign(MATRIX[i]);
      mapper.map(new IntWritable(i), new VectorWritable(v), context);
    }
    
    // check the results
    assertEquals("Number of map results", MATRIX.length, writer.getData().size());
    for (int i = 0; i < MATRIX.length; i++) {
      List<VectorWritable> list = writer.getValue(new IntWritable(i));
      assertEquals("Only one vector per key", 1, list.size());
      Vector v = list.get(0).get();
      for (int j = 0; j < MATRIX[i].length; j++) {
        double total = Math.sqrt(VECTOR[i]) * Math.sqrt(VECTOR[j]) * MATRIX[i][j];
        assertEquals("Product matrix elements", total, v.get(j),EPSILON);
      }
    }
  }
}
