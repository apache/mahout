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
import java.util.Iterator;

public class SequentialAccessSparseVectorWritable extends VectorWritable {

  public SequentialAccessSparseVectorWritable(SequentialAccessSparseVector vector) {
    super(vector);
  }

  public SequentialAccessSparseVectorWritable() {
  }

  @Override
  public void write(DataOutput out) throws IOException {
    SequentialAccessSparseVector sequentialVector = (SequentialAccessSparseVector) get();
    out.writeInt(sequentialVector.size());
    out.writeInt(sequentialVector.getNumNondefaultElements());
    Iterator<Vector.Element> iter = sequentialVector.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      out.writeInt(element.index());
      out.writeDouble(element.get());
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int size = in.readInt();
    int numNonDefaultElements = in.readInt();
    OrderedIntDoubleMapping values = new OrderedIntDoubleMapping(numNonDefaultElements);
    for (int i = 0; i < numNonDefaultElements; i++) {
      int index = in.readInt();
      double value = in.readDouble();
      values.set(index, value);
    }
    set(new SequentialAccessSparseVector(size, values));
  }

}
