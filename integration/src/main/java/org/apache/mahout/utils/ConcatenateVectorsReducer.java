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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.base.Preconditions;

/*
 * Moded combiner/reducer. If vector comes in as length A or length B, concatenated.ˇ
 * If it is length A + B, combiner has already concatenated.
 * 
 */

public class ConcatenateVectorsReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  
  int dimsA = 0;
  int dimsB = 0;
  
  public ConcatenateVectorsReducer() {
    
  }
  
  public void setup(Context context) throws java.io.IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();

    dimsA = Integer.valueOf(configuration.getStrings(ConcatenateVectorsJob.MATRIXA_DIMS)[0]);
    dimsB = Integer.valueOf(configuration.getStrings(ConcatenateVectorsJob.MATRIXB_DIMS)[0]);
  }
  
  public void reduce(IntWritable row, Iterable<VectorWritable> vectorWritableIterable,
                        Context ctx) throws java.io.IOException ,InterruptedException {
    Vector vA = null;
    Vector vB = null;
    Vector vOut = null;
    boolean isNamed = false;
    String name = null;

    for (VectorWritable vw: vectorWritableIterable) {
      Vector v = vw.get();
      if (v instanceof NamedVector) {
        name = ((NamedVector) v).getName();
        isNamed = true;
      }

      if (v.size() == dimsA) {
        vA = v;
      } else if (v.size() == dimsB) {
        vB = v;
      } else if (v.size() == dimsA + dimsB) {
        vOut = v;
        break;
      }
    }

    Preconditions.checkArgument((vA != null || vB != null) || (vOut != null));

    if (vOut == null) {
      vOut = new SequentialAccessSparseVector(dimsA + dimsB);
      if (isNamed) {
        vOut = new NamedVector(vOut, name);
      }
    }

    if (vA != null) {
      appendVector(vOut, vA, 0);
    }

    if (vB != null) {
      appendVector(vOut, vB, dimsA);
    }
    ctx.write(row, new VectorWritable(vOut));
  }
  
  private void appendVector(Vector vOut, Vector vIn, int offset) {
    for (Vector.Element element : vIn.nonZeroes()) {
      vOut.set(element.index() + offset, element.get());
    }
  }
}
