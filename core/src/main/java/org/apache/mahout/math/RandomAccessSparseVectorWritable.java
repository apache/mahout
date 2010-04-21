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

import org.apache.mahout.math.map.OpenIntDoubleHashMap;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

public class RandomAccessSparseVectorWritable extends VectorWritable {

  public RandomAccessSparseVectorWritable(RandomAccessSparseVector vector) {
    super(vector);
  }

  public RandomAccessSparseVectorWritable() {
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    RandomAccessSparseVector randomVector = (RandomAccessSparseVector) get();
    dataOutput.writeInt(randomVector.size());
    dataOutput.writeInt(randomVector.getNumNondefaultElements());
    Iterator<Vector.Element> iter = randomVector.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      dataOutput.writeInt(element.index());
      dataOutput.writeDouble(element.get());
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    int size = dataInput.readInt();
    int numNonDefaultElements = dataInput.readInt();
    OpenIntDoubleHashMap values = new OpenIntDoubleHashMap(numNonDefaultElements);
    for (int i = 0; i < numNonDefaultElements; i++) {
      int index = dataInput.readInt();
      double value = dataInput.readDouble();
      values.put(index, value);
    }
    set(new RandomAccessSparseVector(size, values));
  }

}
