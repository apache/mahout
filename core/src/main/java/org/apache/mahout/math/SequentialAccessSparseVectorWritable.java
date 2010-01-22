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

import org.apache.hadoop.io.Writable;

public class SequentialAccessSparseVectorWritable extends SequentialAccessSparseVector implements Writable {

  public SequentialAccessSparseVectorWritable(SequentialAccessSparseVector vector) {
    super(vector);
  }

  public SequentialAccessSparseVectorWritable() {
    
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(getClass().getName());
    dataOutput.writeUTF(getName() == null ? "" : getName());
    dataOutput.writeInt(size());
    int nde = getNumNondefaultElements();
    dataOutput.writeInt(nde);
    Iterator<Element> iter = iterateNonZero();
    int count = 0;
    while (iter.hasNext()) {
      Element element = iter.next();
      dataOutput.writeInt(element.index());
      dataOutput.writeDouble(element.get());
      count++;
    }
    assert (nde == count);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    final String className = dataInput.readUTF();
    if(className.equals(getClass().getName())) {
      setName(dataInput.readUTF());
    } else {
      setName(className); // we have already read the class name in VectorWritable
    }
    size = dataInput.readInt();
    int nde = dataInput.readInt();
    OrderedIntDoubleMapping values = new OrderedIntDoubleMapping(nde);
    for (int i = 0; i < nde; i++) {
      values.set(dataInput.readInt(), dataInput.readDouble());
    }
    this.values = values;
  }


}
