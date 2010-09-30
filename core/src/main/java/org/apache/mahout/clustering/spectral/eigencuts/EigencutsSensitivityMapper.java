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

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.spectral.common.VectorCache;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

public class EigencutsSensitivityMapper extends
    Mapper<IntWritable, VectorWritable, IntWritable, EigencutsSensitivityNode> {

  private Vector eigenvalues;
  private Vector diagonal;
  private double beta0;
  private double epsilon;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration config = context.getConfiguration();
    beta0 = Double.parseDouble(config.get(EigencutsKeys.BETA));
    epsilon = Double.parseDouble(config.get(EigencutsKeys.EPSILON));
    
    // read in the two vectors from the cache
    eigenvalues = VectorCache.load(
        new IntWritable(EigencutsKeys.EIGENVALUES_CACHE_INDEX), config);
    diagonal = VectorCache.load(
        new IntWritable(EigencutsKeys.DIAGONAL_CACHE_INDEX), config);
    if (!(eigenvalues instanceof SequentialAccessSparseVector || eigenvalues instanceof DenseVector)) {
      eigenvalues = new SequentialAccessSparseVector(eigenvalues);
    }
    if (!(diagonal instanceof SequentialAccessSparseVector || diagonal instanceof DenseVector)) {
      diagonal = new SequentialAccessSparseVector(diagonal);
    }
  }
  
  @Override
  protected void map(IntWritable row, VectorWritable vw, Context context) 
    throws IOException, InterruptedException {
    
    // first, does this particular eigenvector even pass the required threshold?
    double eigenvalue = Math.abs(eigenvalues.get(row.get()));
    double betak = -Functions.LOGARITHM.apply(2) / Functions.LOGARITHM.apply(eigenvalue);
    if (eigenvalue >= 1.0 || betak <= (epsilon * beta0)) {
      // doesn't pass the threshold! quit
      return;
    }
    
    // go through the vector, performing the calculations
    // sadly, no way to get around n^2 computations      
    Map<Integer, EigencutsSensitivityNode> columns = new HashMap<Integer, EigencutsSensitivityNode>();
    Vector ev = vw.get();
    for (int i = 0; i < ev.size(); i++) {
      double minS_ij = Double.MAX_VALUE;
      int minInd = -1;
      for (int j = 0; j < ev.size(); j++) {          
        double S_ij = performSensitivityCalculation(eigenvalue, ev.get(i), 
            ev.get(j), diagonal.get(i), diagonal.get(j));
        
        // perform non-maximal suppression
        // is this the smallest value in the row?
        if (S_ij < minS_ij) {
          minS_ij = S_ij;
          minInd = j;
        }
      }
      
      // is this the smallest value in the column?
      Integer column = new Integer(minInd);
      EigencutsSensitivityNode value = new EigencutsSensitivityNode(i, minInd, minS_ij);
      if (!columns.containsKey(column)) {
        columns.put(column, value);
      } else if (columns.get(column).getSensitivity() > minS_ij) {
        columns.remove(column);
        columns.put(column, value);
      }
    }
    
    // write whatever values made it through
    
    for (EigencutsSensitivityNode e : columns.values().toArray(new EigencutsSensitivityNode[0])) {
      context.write(new IntWritable(e.getRow()), e);
    }
  }
  
  /**
   * Helper method, performs the actual calculation. Looks something like this:
   *
   * (log(2) / lambda_k * log(lambda_k) * log(lambda_k^beta0 / 2)) * [
   * - (((u_i / sqrt(d_i)) - (u_j / sqrt(d_j)))^2 + (1 - lambda) * 
   *   ((u_i^2 / d_i) + (u_j^2 / d_j))) ]
   * 
   * @param eigenvalue
   * @param ev_i
   * @param ev_j
   * @param diag_i
   * @param diag_j
   * @return
   */
  private double performSensitivityCalculation(double eigenvalue, double
      ev_i, double ev_j, double diag_i, double diag_j) {
    
    double firsthalf = Functions.LOGARITHM.apply(2) / (
        eigenvalue * Functions.LOGARITHM.apply(eigenvalue) * 
        Functions.LOGARITHM.apply(Functions.POW.apply(eigenvalue, beta0) / 2));
    
    double secondhalf = -Functions.POW.apply(((ev_i / 
        Functions.SQRT.apply(diag_i)) - (ev_j / 
        Functions.SQRT.apply(diag_j))), 2) + (1 - eigenvalue) * 
        ((Functions.POW.apply(ev_i, 2) / diag_i) + 
        (Functions.POW.apply(ev_j, 2) / diag_j));
    
    return firsthalf * secondhalf;
  }
  
  /**
   * Utility helper method, used for unit testing.
   * @param beta0
   * @param epsilon
   * @param eigenvalues
   * @param diagonal
   */
  void setup(double beta0, double epsilon, Vector eigenvalues, Vector diagonal) {
    this.beta0 = beta0;
    this.epsilon = epsilon;
    this.eigenvalues = eigenvalues;
    this.diagonal = diagonal;
  }
}
