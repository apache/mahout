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
import org.apache.mahout.clustering.spectral.UnitVectorizerJob.UnitVectorizerMapper;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

public class TestUnitVectorizerJob extends MahoutTestCase {

  private static final double [][] RAW = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

  @Test
  public void testUnitVectorizerMapper() throws Exception {
    UnitVectorizerMapper mapper = new UnitVectorizerMapper();
    Configuration conf = getConfiguration();
    
    // set up the dummy writers
    DummyRecordWriter<IntWritable, VectorWritable> writer = new
      DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context 
      context = DummyRecordWriter.build(mapper, conf, writer);
    
    // perform the mapping
    for (int i = 0; i < RAW.length; i++) {
      Vector vector = new RandomAccessSparseVector(RAW[i].length);
      vector.assign(RAW[i]);
      mapper.map(new IntWritable(i), new VectorWritable(vector), context);
    }
    
    // check the results
    assertEquals("Number of map results", RAW.length, writer.getData().size());
    for (int i = 0; i < RAW.length; i++) {
      IntWritable key = new IntWritable(i);
      List<VectorWritable> list = writer.getValue(key);
      assertEquals("Only one element per row", 1, list.size());
      Vector v = list.get(0).get();
      assertTrue("Unit vector sum is 1 or differs by 0.0001", Math.abs(v.norm(2) - 1) < 0.000001);
    }
  } 
}
