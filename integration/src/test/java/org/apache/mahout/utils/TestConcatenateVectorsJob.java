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

package org.apache.mahout.utils;

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * Code stolen from TestAffinityMatrixJob. Like TAMJ, it tests the Mappers/Reducers but not test the job
 */

public class TestConcatenateVectorsJob extends MahoutTestCase {
  
  private static final double [][] DATA_A = {
    {0,1,2,3,4},
    {},
    {0,1,2,3,4}
  };
  private static final double [][] DATA_B = {
    {},
    {5,6,7},
    {5,6,7}
  };
  
  @Test
  public void testConcatenateVectorsReducer() throws Exception {
    
    Configuration configuration = getConfiguration();
    configuration.set(ConcatenateVectorsJob.MATRIXA_DIMS, "5");
    configuration.set(ConcatenateVectorsJob.MATRIXB_DIMS, "3");
    
    // Yes, all of this generic rigmarole is needed, and woe betide he who changes it
    ConcatenateVectorsReducer reducer = new ConcatenateVectorsReducer();

    DummyRecordWriter<IntWritable, VectorWritable> recordWriter = new DummyRecordWriter<IntWritable, VectorWritable>();

    Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context reduceContext =
      DummyRecordWriter.build(reducer, configuration, recordWriter, IntWritable.class, VectorWritable.class);
    
    reducer.setup(reduceContext);
    
    for(int i = 0; i < 3; i++) {
      double[] values = DATA_A[i];
      List<VectorWritable> vwList = Lists.newArrayList();
      if (values.length > 0) {
        Vector v = new DenseVector(values);
        VectorWritable vw = new VectorWritable();
        vw.set(v);
        vwList.add(vw);
      }
      values = DATA_B[i];
      if (values.length > 0) {
        Vector v = new DenseVector(values);
        VectorWritable vw = new VectorWritable();
        vw.set(v);
        vwList.add(vw);

      }
      IntWritable row = new IntWritable(i);
      
      reducer.reduce(row, vwList, reduceContext);
    }
    
    for (IntWritable row : recordWriter.getKeys()) {
      List<VectorWritable> list = recordWriter.getValue(row);
      Vector v = list.get(0).get();
      assertEquals(8, v.size());
      for (Vector.Element element : v.nonZeroes()) {
        assertEquals(element.index(), v.get(element.index()), 0.001);
      }
    }
  }
  
}
