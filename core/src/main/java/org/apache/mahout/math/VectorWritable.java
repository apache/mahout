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

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

public class VectorWritable extends Configured implements Writable {

  public static final int FLAG_DENSE = 0x01;
  public static final int FLAG_SEQUENTIAL = 0x02;
  public static final int FLAG_NAMED = 0x04;
  public static final int NUM_FLAGS = 3;

  private Vector vector;

  public VectorWritable() {
  }

  public VectorWritable(Vector vector) {
    this.vector = vector;
  }

  public Vector get() {
    return vector;
  }

  public void set(Vector vector) {
    this.vector = vector;
  }

  @Override
  public void write(DataOutput out) throws IOException {

    boolean dense = vector.isDense();
    boolean sequential = vector.isSequentialAccess();
    boolean named = vector instanceof NamedVector;

    int flags = (dense ? FLAG_DENSE : 0) | (sequential ? FLAG_SEQUENTIAL : 0) | (named ? FLAG_NAMED : 0);
    out.writeByte(flags);

    if (dense) {
      out.writeInt(vector.size());
      for (Vector.Element element : vector) {
        out.writeDouble(element.get());
      }
    } else {
      out.writeInt(vector.size());
      out.writeInt(vector.getNumNondefaultElements());
      Iterator<Vector.Element> iter = vector.iterateNonZero();
      while (iter.hasNext()) {
        Vector.Element element = iter.next();
        out.writeInt(element.index());
        out.writeDouble(element.get());
      }
    }
    if (named) {
      out.writeUTF(((NamedVector) vector).getName());
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int flags = in.readByte();
    if (flags >> NUM_FLAGS != 0) {
      throw new IllegalArgumentException();
    }
    boolean dense = (flags & FLAG_DENSE) != 0;
    boolean sequential = (flags & FLAG_SEQUENTIAL) != 0;
    boolean named = (flags & FLAG_NAMED) != 0;

    Vector v;
    if (dense) {
      int size = in.readInt();
      double[] values = new double[size];
      for (int i = 0; i < size; i++) {
        values[i] = in.readDouble();
      }
      v = new DenseVector(values);
    } else {
      int size = in.readInt();
      int numNonDefaultElements = in.readInt();
      if (sequential) {
        v = new SequentialAccessSparseVector(size, numNonDefaultElements);
      } else {
        v = new RandomAccessSparseVector(size, numNonDefaultElements);
      }
      for (int i = 0; i < numNonDefaultElements; i++) {
        int index = in.readInt();
        double value = in.readDouble();
        v.setQuick(index, value);
      }
    }
    if (named) {
      String name = in.readUTF();
      v = new NamedVector(v, name);
    }
    vector = v;
  }

  /** Write the vector to the output */
  public static void writeVector(DataOutput out, Vector vector) throws IOException {
    new VectorWritable(vector).write(out);
  }

}
