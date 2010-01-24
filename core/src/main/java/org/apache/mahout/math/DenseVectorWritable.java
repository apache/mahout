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

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

public class DenseVectorWritable extends DenseVector implements Writable {

  public DenseVectorWritable() {
    
  }

  public DenseVectorWritable(DenseVector v) {
    setName(v.getName());
    values = v.values;
    lengthSquared = v.lengthSquared;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(getClass().getName());
    dataOutput.writeUTF(this.getName() == null ? "" : this.getName());
    dataOutput.writeInt(size());
    dataOutput.writeDouble(lengthSquared);
    Iterator<Vector.Element> iter = iterateAll();
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      dataOutput.writeDouble(element.get());
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    String className = dataInput.readUTF();
    if(className.equals(getClass().getName())) {
      this.setName(dataInput.readUTF());
    } else {
      setName(className); // we have already read the class name in VectorWritable
    }
    double[] values = new double[dataInput.readInt()];
    lengthSquared = dataInput.readDouble();
    for (int i = 0; i < values.length; i++) {
      values[i] = dataInput.readDouble();
    }
    this.values = values;
  }
  
}
