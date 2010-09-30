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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * <p>Tests the Eigencuts M/R process for generating perturbation sensitivities
 * in a weighted graph.</p>
 * 
 * <p>This process requires a lot of inputs. Please read the 
 * EigencutsSensitivityJob javadocs for more information on these variables.
 * For now, 
 *
 */
public class TestEigencutsSensitivityJob extends MahoutTestCase {
  
  /*
  private final double [][] affinity = { {0, 0.9748, 0.6926, 0.6065},
                                         {0.9748, 0, 0.7178, 0.6350},
                                         {0.6926, 0.7178, 0, 0.9898},
                                         {0.6065, 0.6350, 0.9898, 0} };
  */
  private final double [] diagonal = {2.2739, 2.3276, 2.4002, 2.2313};
  
  private final double [][] eigenvectors = { {-0.4963, -0.5021, -0.5099, -0.4916},
                                             {-0.5143, -0.4841, 0.4519, 0.5449},
                                             {-0.6858, 0.7140, -0.1146, 0.0820},
                                             {0.1372, -0.0616, -0.7230, 0.6743} };
  private final double [] eigenvalues = {1.000, -0.1470, -0.4238, -0.4293};

  /**
   * This is the toughest step, primarily because of the intensity of 
   * the calculations that are performed and the amount of data required.
   * Four parameters in particular - the list of eigenvalues, the 
   * vector representing the diagonal matrix, and the scalars beta0 and 
   * epsilon - must be set here prior to the start of the mapper. Once
   * the mapper is executed, it iterates over a matrix of all corresponding
   * eigenvectors.
   * @throws Exception
   */
@Test
public void testEigencutsSensitivityMapper() throws Exception {
    EigencutsSensitivityMapper mapper = new EigencutsSensitivityMapper();
    Configuration conf = new Configuration();

    // construct the writers
    DummyRecordWriter<IntWritable, EigencutsSensitivityNode> writer = 
      new DummyRecordWriter<IntWritable, EigencutsSensitivityNode>();
    Mapper<IntWritable, VectorWritable, IntWritable, EigencutsSensitivityNode>.Context 
      context = DummyRecordWriter.build(mapper, conf, writer);
    mapper.setup(2.0, 0.25, new DenseVector(eigenvalues), new DenseVector(diagonal));
    
    // perform the mapping
    for (int i = 0; i < eigenvectors.length; i++) {
      VectorWritable row = new VectorWritable(new DenseVector(eigenvectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    
    // the results line up
    for (IntWritable key : writer.getKeys()) {
      List<EigencutsSensitivityNode> list = writer.getValue(key);
      assertEquals("Only one result per row", 1, list.size());
      EigencutsSensitivityNode item = list.get(0);
      assertTrue("Sensitivity values are correct", Math.abs(item.getSensitivity() + 0.48) < 0.01);
    }
  }
  
  /**
   * This step will simply assemble sensitivities into one coherent matrix.
   * @throws Exception
   */
@Test
  public void testEigencutsSensitivityReducer() throws Exception {
    EigencutsSensitivityMapper mapper = new EigencutsSensitivityMapper();
    Configuration conf = new Configuration();
    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, eigenvectors.length);
    
    // construct the writers
    DummyRecordWriter<IntWritable, EigencutsSensitivityNode> mapWriter = 
      new DummyRecordWriter<IntWritable, EigencutsSensitivityNode>();
    Mapper<IntWritable, VectorWritable, IntWritable, EigencutsSensitivityNode>.Context 
      mapContext = DummyRecordWriter.build(mapper, conf, mapWriter);
    mapper.setup(2.0, 0.25, new DenseVector(eigenvalues), new DenseVector(diagonal));
    
    // perform the mapping
    for (int i = 0; i < eigenvectors.length; i++) {
      VectorWritable row = new VectorWritable(new DenseVector(eigenvectors[i]));
      mapper.map(new IntWritable(i), row, mapContext);
    }
    
    // set up the values for the reducer
    conf.set(EigencutsKeys.DELTA, "1.0");
    conf.set(EigencutsKeys.TAU, "-0.1");
    
    EigencutsSensitivityReducer reducer = new EigencutsSensitivityReducer();
    // set up the writers
    DummyRecordWriter<IntWritable, VectorWritable> redWriter = new
      DummyRecordWriter<IntWritable, VectorWritable>();
    Reducer<IntWritable, EigencutsSensitivityNode, IntWritable, VectorWritable>.Context
      redContext = DummyRecordWriter.build(reducer, conf, redWriter, 
      IntWritable.class, EigencutsSensitivityNode.class);
    
    // perform the reduction
    for (IntWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }
    
    // since all the sensitivities were below the threshold,
    // each of them should have survived
    for (IntWritable key : redWriter.getKeys()) {
      List<VectorWritable> list = redWriter.getValue(key);
      assertEquals("One item in the list", 1, list.size());
      Vector item = list.get(0).get();
      
      // should only be one non-zero item
      assertTrue("One non-zero item in the array", Math.abs(item.zSum() + 0.48) < 0.01);
    }

  }
}
