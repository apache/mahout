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
package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;

/**
 * Ad-hoc substitution for {@link org.apache.mahout.math.MatrixWritable}.
 * Perhaps more useful for situations with mostly dense data (such as Q-blocks)
 * but reduces GC by reusing the same block memory between loads and writes.
 * <p>
 * 
 * in case of Q blocks, it doesn't even matter if they this data is dense cause
 * we need to unpack it into dense for fast access in computations anyway and
 * even if it is not so dense the block compressor in sequence files will take
 * care of it for the serialized size.
 * <p>
 */
public class DenseBlockWritable implements Writable {
  private double[][] block;

  public void setBlock(double[][] block) {
    this.block = block;
  }

  public double[][] getBlock() {
    return block;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int m = in.readInt();
    int n = in.readInt();
    if (block == null) {
      block = new double[m][0];
    } else if (block.length != m) {
      block = Arrays.copyOf(block, m);
    }
    for (int i = 0; i < m; i++) {
      if (block[i] == null || block[i].length != n) {
        block[i] = new double[n];
      }
      for (int j = 0; j < n; j++) {
        block[i][j] = in.readDouble();
      }

    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    int m = block.length;
    int n = block.length == 0 ? 0 : block[0].length;

    out.writeInt(m);
    out.writeInt(n);
    for (double[] aBlock : block) {
      for (int j = 0; j < n; j++) {
        out.writeDouble(aBlock[j]);
      }
    }
  }

}
