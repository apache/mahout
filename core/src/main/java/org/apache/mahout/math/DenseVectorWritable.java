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

package org.apache.mahout.math;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class DenseVectorWritable extends VectorWritable {

  public DenseVectorWritable(DenseVector vector) {
    super(vector);
  }

  public DenseVectorWritable() {
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    DenseVector denseVector = (DenseVector) get();
    dataOutput.writeInt(denseVector.size());
    dataOutput.writeDouble(denseVector.getLengthSquared());
    for (Vector.Element element : denseVector) {
      dataOutput.writeDouble(element.get());
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int size = in.readInt();
    double[] values = new double[size];
    double lengthSquared = in.readDouble();
    for (int i = 0; i < values.length; i++) {
      values[i] = in.readDouble();
    }
    set(new DenseVector(values));
  }
  
}
