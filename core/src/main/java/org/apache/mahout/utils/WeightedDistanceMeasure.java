/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.matrix.*;

import java.io.DataInputStream;
import java.io.FileNotFoundException;

/**
 * Abstract implementation of DistanceMeasure with support for weights.
 */
public abstract class WeightedDistanceMeasure extends AbstractDistanceMeasure {

  protected Vector weights;

  /**
   * If existing, loads weights using a SparseVectorWritable
   * from file set in jobConf parameter "org.apache.mahout.utils.WeightedDistanceMeasure.sparseVector"
   *
   * todo: should be able to handle any sort of vector. perhaps start the file with what class it is?
   * todo: some nice static helper method to write and read the file,
   * todo: or should it be a new writable that decorates any given vector?
   *
   * @param jobConf
   */
  public void configure(JobConf jobConf) {
    try {
      FileSystem fs = FileSystem.get(jobConf);
      String weightsPathName = WeightedDistanceMeasure.class.getName() + ".sparseVector";
      if (weightsPathName != null) {
        Vector weights = new SparseVector();
        Path weightsPath = new Path(weightsPathName);
        if (!fs.exists(weightsPath)) {
          throw new FileNotFoundException(weightsPath.toString());
        }
        DataInputStream in = fs.open(weightsPath);
        weights.readFields(in);
        in.close();
        this.weights = weights;
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public Vector getWeights() {
    return weights;
  }

  public void setWeights(Vector weights) {
    this.weights = weights;
  }




}
