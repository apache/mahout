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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.spectral.common.VertexWritable;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsAffinityCutsJob.EigencutsAffinityCutsCombiner;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsAffinityCutsJob.EigencutsAffinityCutsMapper;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsAffinityCutsJob.EigencutsAffinityCutsReducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * <p>Tests the Eigencuts affinity matrix "cut" ability, the core functionality
 * of the algorithm responsible for making the clusterings.</p>
 * 
 * <p>Due to the complexity of this section, and the amount of data required,
 * there are three steps: the mapper essentially reads in the affinity/cut
 * matrices and creating "vertices" of points, the combiner performs the 
 * actual checks on the sensitivities and zeroes out the necessary affinities,
 * and at last the reducer reforms the affinity matrix.</p>
 */
public class TestEigencutsAffinityCutsJob extends MahoutTestCase {
  
  private final double [][] affinity = { {0, 10, 2, 1}, {10, 0, 2, 2},
                                  {2, 2, 0, 10}, {1, 2, 10, 0} };
  private final double [][] sensitivity = { {0, 0, 1, 1}, {0, 0, 1, 1},
                                {1, 1, 0, 0}, {1, 1, 0, 0} };

  /**
   * Testing the mapper is fairly straightforward: there are two matrices
   * to be processed simultaneously (cut matrix of sensitivities, and the
   * affinity matrix), and since both are symmetric, two entries from each
   * will be grouped together with the same key (or, in the case of an
   * entry along the diagonal, only two entries).
   * 
   * The correct grouping of these quad or pair vertices is the only
   * output of the mapper.
   * 
   * @throws Exception
   */
  @Test
  public void testEigencutsAffinityCutsMapper() throws Exception {
    EigencutsAffinityCutsMapper mapper = new EigencutsAffinityCutsMapper();
    Configuration conf = new Configuration();
    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, this.affinity.length);
    
    // set up the writer
    DummyRecordWriter<Text, VertexWritable> writer = 
      new DummyRecordWriter<Text, VertexWritable>();
    Mapper<IntWritable, VectorWritable, Text, VertexWritable>.Context context = 
      DummyRecordWriter.build(mapper, conf, writer);
    
    // perform the maps
    for (int i = 0; i < this.affinity.length; i++) {
      VectorWritable aff = new VectorWritable(new DenseVector(this.affinity[i]));
      VectorWritable sens = new VectorWritable(new DenseVector(this.sensitivity[i]));
      IntWritable key = new IntWritable(i);
      mapper.map(key, aff, context);
      mapper.map(key, sens, context);
    }
    
    // were the vertices constructed correctly? if so, then for two 4x4
    // matrices, there should be 10 unique keys with 56 total entries
    assertEquals("Number of keys", 10, writer.getKeys().size());
    for (int i = 0; i < this.affinity.length; i++) {
      for (int j = 0; j < this.affinity.length; j++) {
        Text key = new Text(Math.max(i, j) + "_" + Math.min(i,j));
        List<VertexWritable> values = writer.getValue(key);
        
        // if we're on a diagonal, there should only be 2 entries
        // otherwise, there should be 4
        if (i == j) {
          assertEquals("Diagonal entry", 2, values.size());
          for (VertexWritable v : values) {
            assertFalse("Diagonal values are zero", v.getValue() > 0);
          }
        } else {
          assertEquals("Off-diagonal entry", 4, values.size());
          if (i + j == 3) { // all have values greater than 0
            for (VertexWritable v : values) {
              assertTrue("Off-diagonal non-zero entries", v.getValue() > 0);
            }
          }
        }
      }
    }
  }
  
  /**
   * This is by far the trickiest step. However, an easy condition is if 
   * we have only two vertices - indicating vertices on the diagonal of the
   * two matrices - then we simply exit (since the algorithm does not operate
   * on the diagonal; it makes no sense to perform cuts by isolating data
   * points from themselves).
   * 
   * If there are four points, then first we must separate the two which
   * belong to the affinity matrix from the two that are sensitivities. In theory,
   * each pair should have exactly the same value (symmetry). If the sensitivity
   * is below a certain threshold, then we set the two values of the affinity
   * matrix to 0 (but not before adding the affinity values to the diagonal, so
   * as to maintain the overall sum of the row of the affinity matrix).
   * 
   * @throws Exception
   */
  @Test
  public void testEigencutsAffinityCutsCombiner() throws Exception {
    Configuration conf = new Configuration();
    Path affinity = new Path("affinity");
    Path sensitivity = new Path("sensitivity");
    conf.set(EigencutsKeys.AFFINITY_PATH, affinity.getName());
    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, this.affinity.length);
    
    // since we need the working paths to distinguish the vertex types, 
    // we can't use the mapper (since we have no way of manually setting
    // the Context.workingPath() )
    Map<Text, List<VertexWritable>> data = buildMapData(affinity, sensitivity, this.sensitivity);
     
    // now, set up the combiner
    EigencutsAffinityCutsCombiner combiner = new EigencutsAffinityCutsCombiner();
    DummyRecordWriter<Text, VertexWritable> redWriter =
      new DummyRecordWriter<Text, VertexWritable>();
    Reducer<Text, VertexWritable, Text, VertexWritable>.Context 
      redContext = DummyRecordWriter.build(combiner, conf, redWriter, Text.class,
      VertexWritable.class);
    
    // perform the combining
    for (Map.Entry<Text, List<VertexWritable>> entry : data.entrySet()) {
      combiner.reduce(entry.getKey(), entry.getValue(), redContext);
    }
    
    // test the number of cuts, there should be 2
    assertEquals("Number of cuts detected", 4, 
        redContext.getCounter(EigencutsAffinityCutsJob.CUTSCOUNTER.NUM_CUTS).getValue());
    
    // loop through all the results; let's see if they match up to our
    // affinity matrix (and all the cuts appear where they should
    Map<Text, List<VertexWritable>> results = redWriter.getData();
    for (Map.Entry<Text, List<VertexWritable>> entry : results.entrySet()) {
      List<VertexWritable> row = entry.getValue();
      IntWritable key = new IntWritable(Integer.parseInt(entry.getKey().toString()));
      
      double calcDiag = 0.0;
      double trueDiag = sumOfRowCuts(key.get(), this.sensitivity);
      for (VertexWritable e : row) {

        // should the value have been cut, e.g. set to 0?
        if (key.get() == e.getCol()) {
          // we have our diagonal
          calcDiag += e.getValue();
        } else if (this.sensitivity[key.get()][e.getCol()] == 0.0) {
          // no, corresponding affinity should have same value as before
          assertEquals("Preserved affinity value", 
              this.affinity[key.get()][e.getCol()], e.getValue(),EPSILON);
        } else {
          // yes, corresponding affinity value should be 0
          assertEquals("Cut affinity value", 0.0, e.getValue(),EPSILON);
        }
      }
      // check the diagonal has the correct sum
      assertEquals("Diagonal sum from cuts", trueDiag, calcDiag,EPSILON);
    }
  }
  
  /**
   * Fairly straightforward: the task here is to reassemble the rows of the
   * affinity matrix. The tricky part is that any specific element in the list
   * of elements which does NOT lay on the diagonal will be so because it
   * did not drop below the sensitivity threshold, hence it was not "cut". 
   * 
   * On the flip side, there will be many entries whose coordinate is now
   * set to the diagonal, indicating they were previously affinity entries
   * whose sensitivities were below the threshold, and hence were "cut" - 
   * set to 0 at their original coordinates, and had their values added to
   * the diagonal entry (hence the numerous entries with the coordinate of
   * the diagonal).
   * 
   * @throws Exception
   */
  @Test
  public void testEigencutsAffinityCutsReducer() throws Exception {
    Configuration conf = new Configuration();
    Path affinity = new Path("affinity");
    Path sensitivity = new Path("sensitivity");
    conf.set(EigencutsKeys.AFFINITY_PATH, affinity.getName());
    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, this.affinity.length);
    
    // since we need the working paths to distinguish the vertex types, 
    // we can't use the mapper (since we have no way of manually setting
    // the Context.workingPath() )
    Map<Text, List<VertexWritable>> data = buildMapData(affinity, sensitivity, this.sensitivity);
     
    // now, set up the combiner
    EigencutsAffinityCutsCombiner combiner = new EigencutsAffinityCutsCombiner();
    DummyRecordWriter<Text, VertexWritable> comWriter =
      new DummyRecordWriter<Text, VertexWritable>();
    Reducer<Text, VertexWritable, Text, VertexWritable>.Context 
      comContext = DummyRecordWriter.build(combiner, conf, comWriter, Text.class,
      VertexWritable.class);
    
    // perform the combining
    for (Map.Entry<Text, List<VertexWritable>> entry : data.entrySet()) {
      combiner.reduce(entry.getKey(), entry.getValue(), comContext);
    }
    
    // finally, set up the reduction writers
    EigencutsAffinityCutsReducer reducer = new EigencutsAffinityCutsReducer();
    DummyRecordWriter<IntWritable, VectorWritable> redWriter = new
      DummyRecordWriter<IntWritable, VectorWritable>();
    Reducer<Text, VertexWritable, IntWritable, VectorWritable>.Context 
      redContext = DummyRecordWriter.build(reducer, conf, redWriter, 
      Text.class, VertexWritable.class);
    
    // perform the reduction
    for (Text key : comWriter.getKeys()) {
      reducer.reduce(key, comWriter.getValue(key), redContext);
    }
    
    // now, check that the affinity matrix is correctly formed
    for (IntWritable row : redWriter.getKeys()) {
      List<VectorWritable> results = redWriter.getValue(row);
      // there should only be 1 vector
      assertEquals("Only one vector with a given row number", 1, results.size());
      Vector therow = results.get(0).get();
      for (Vector.Element e : therow) {
        // check the diagonal
        if (row.get() == e.index()) {
          assertEquals("Correct diagonal sum of cuts", sumOfRowCuts(row.get(), 
              this.sensitivity), e.get(),EPSILON);
        } else {
          // not on the diagonal...if it was an element labeled to be cut,
          // it should have a value of 0. Otherwise, it should have kept its
          // previous value
          if (this.sensitivity[row.get()][e.index()] == 0.0) {
            // should be what it was originally
            assertEquals("Preserved element", this.affinity[row.get()][e.index()], e.get(), EPSILON);
          } else {
            // should be 0
            assertEquals("Cut element", 0.0, e.get(), EPSILON);
          }
        }
      }
    }
  }
  
  /**
   * Utility method for simulating the Mapper behavior.
   * @param affinity
   * @param sensitivity
   * @param array
   * @return
   */
  private Map<Text, List<VertexWritable>> buildMapData(Path affinity, 
      Path sensitivity, double [][] array) {
    Map<Text, List<VertexWritable>> map = Maps.newHashMap();
    for (int i = 0; i < this.affinity.length; i++) {
      for (int j = 0; j < this.affinity[i].length; j++) {
        Text key = new Text(Math.max(i, j) + "_" + Math.min(i, j));
        List<VertexWritable> toAdd = Lists.newArrayList();
        if (map.containsKey(key)) {
          toAdd = map.get(key);
          map.remove(key);
        }
        toAdd.add(new VertexWritable(i, j, this.affinity[i][j], affinity.getName()));
        toAdd.add(new VertexWritable(i, j, array[i][j], sensitivity.getName()));
        map.put(key, toAdd);
      }
    }
    return map;
  }
  
  /**
   * Utility method for calculating the new diagonal on the specified row of the
   * affinity matrix after a single iteration, given the specified cut matrix
   * @param row
   * @param cuts
   * @return
   */
  private double sumOfRowCuts(int row, double [][] cuts) {
    double retval = 0.0;
    for (int j = 0; j < this.affinity[row].length; j++) {
      if (cuts[row][j] != 0.0) {
        retval += this.affinity[row][j];
      }
    }
    return retval;
  }
}
