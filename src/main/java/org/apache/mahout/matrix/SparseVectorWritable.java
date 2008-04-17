package org.apache.mahout.matrix;

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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;


public class SparseVectorWritable implements VectorWritable {

  private SparseVector vector;

  public SparseVectorWritable() {
  }

  public SparseVectorWritable(Vector vector) {
    set(vector);
  }

  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(vector.cardinality());
    dataOutput.writeInt(vector.size());
    for (Vector.Element element : vector) {
      if (element.get() != 0d) {
        dataOutput.writeInt(element.index());
        dataOutput.writeDouble(element.get());
      }
    }
  }

  public void readFields(DataInput dataInput) throws IOException {
    int cardinality = dataInput.readInt();
    vector = new SparseVector(cardinality);    
    int size = dataInput.readInt();
    for (int i = 0; i < size; i++) {
      vector.set(dataInput.readInt(), dataInput.readDouble());
    }
  }

  public SparseVector get() {
    return vector;
  }


  public void set(Vector vector) {
    if (vector == null) {
      this.vector = null;
    } else {
      this.vector = new SparseVector(vector.cardinality());
      for (Vector.Element e : vector) {
        this.vector.set(e.index(), e.get());
      }
    }
  }

  public void set(SparseVector vector) {
    this.vector = vector;
  }
}
