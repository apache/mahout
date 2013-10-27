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
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.junit.Test;

/**
 * <p>Tests the affinity matrix input M/R task.</p>
 * 
 * <p>The tricky item with this task is that the format of the input
 * must be correct; it must take the form of a graph input, and for the
 * current implementation, the input must be symmetric, e.g. the weight
 * from node A to B = the weight from node B to A. This is not explicitly
 * enforced within the task itself (since, as of the time these tests were 
 * written, we have not yet decided on a final rule regarding the 
 * symmetry/non-symmetry of the affinity matrix, so we are unofficially 
 * enforcing symmetry). Input looks something like this:</p>
 * 
 * <pre>0, 0, 0
 * 0, 1, 10
 * 0, 2, 20
 * ...
 * 1, 0, 10
 * 2, 0, 20
 * ...</pre>
 * 
 * <p>The mapper's task is simply to convert each line of text into a
 * DistributedRowMatrix entry, allowing the reducer to join each entry
 * of the same row into a VectorWritable.</p>
 * 
 * <p>Exceptions are thrown in cases of bad input format: if there are
 * more or fewer than 3 numbers per line, or any of the numbers are missing.
 */
public class TestAffinityMatrixInputJob extends MahoutTestCase {
  
  private static final String [] RAW = {"0,0,0", "0,1,5", "0,2,10", "1,0,5", "1,1,0",
                                        "1,2,20", "2,0,10", "2,1,20", "2,2,0"};
  private static final int RAW_DIMENSIONS = 3;

  @Test
  public void testAffinityMatrixInputMapper() throws Exception {
    AffinityMatrixInputMapper mapper = new AffinityMatrixInputMapper();
    Configuration conf = getConfiguration();
    conf.setInt(Keys.AFFINITY_DIMENSIONS, RAW_DIMENSIONS);
    
    // set up the dummy writer and the M/R context
    DummyRecordWriter<IntWritable, MatrixEntryWritable> writer =
      new DummyRecordWriter<IntWritable, MatrixEntryWritable>();
    Mapper<LongWritable, Text, IntWritable, MatrixEntryWritable>.Context 
      context = DummyRecordWriter.build(mapper, conf, writer);

    // loop through all the points and test each one is converted
    // successfully to a DistributedRowMatrix.MatrixEntry
    for (String s : RAW) {
      mapper.map(new LongWritable(), new Text(s), context);
    }

    // test the data was successfully constructed
    assertEquals("Number of map results", RAW_DIMENSIONS, writer.getData().size());
    Set<IntWritable> keys = writer.getData().keySet();
    for (IntWritable i : keys) {
      List<MatrixEntryWritable> row = writer.getData().get(i);
      assertEquals("Number of items in row", RAW_DIMENSIONS, row.size());
    }
  }
  
  @Test
  public void testAffinitymatrixInputReducer() throws Exception {
    AffinityMatrixInputMapper mapper = new AffinityMatrixInputMapper();
    Configuration conf = getConfiguration();
    conf.setInt(Keys.AFFINITY_DIMENSIONS, RAW_DIMENSIONS);
    
    // set up the dummy writer and the M/R context
    DummyRecordWriter<IntWritable, MatrixEntryWritable> mapWriter =
      new DummyRecordWriter<IntWritable, MatrixEntryWritable>();
    Mapper<LongWritable, Text, IntWritable, MatrixEntryWritable>.Context
      mapContext = DummyRecordWriter.build(mapper, conf, mapWriter);

    // loop through all the points and test each one is converted
    // successfully to a DistributedRowMatrix.MatrixEntry
    for (String s : RAW) {
      mapper.map(new LongWritable(), new Text(s), mapContext);
    }
    // store the data for checking later
    Map<IntWritable, List<MatrixEntryWritable>> map = mapWriter.getData();

    // now reduce the data
    AffinityMatrixInputReducer reducer = new AffinityMatrixInputReducer();
    DummyRecordWriter<IntWritable, VectorWritable> redWriter = 
      new DummyRecordWriter<IntWritable, VectorWritable>();
    Reducer<IntWritable, MatrixEntryWritable,
      IntWritable, VectorWritable>.Context redContext = DummyRecordWriter
      .build(reducer, conf, redWriter, IntWritable.class, MatrixEntryWritable.class);
    for (IntWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }
    
    // check that all the elements are correctly ordered
    assertEquals("Number of reduce results", RAW_DIMENSIONS, redWriter.getData().size());
    for (IntWritable row : redWriter.getKeys()) {
      List<VectorWritable> list = redWriter.getValue(row);
      assertEquals("Should only be one vector", 1, list.size());
      // check that the elements in the array are correctly ordered
      Vector v = list.get(0).get();
      for (Vector.Element e : v.all()) {
        // find this value in the original map
        MatrixEntryWritable toCompare = new MatrixEntryWritable();
        toCompare.setRow(-1);
        toCompare.setCol(e.index());
        toCompare.setVal(e.get());
        assertTrue("This entry was correctly placed in its row", map.get(row).contains(toCompare));
      }
    }
  }
}
