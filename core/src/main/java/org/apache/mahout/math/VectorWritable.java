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

public class VectorWritable extends Configured implements Writable {

  private Vector vector;

  public Vector get() {
    return vector;
  }

  public void set(Vector vector) {
    this.vector = vector;
  }

  public VectorWritable() {
  }

  public VectorWritable(Vector v) {
    vector = v;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Writable w;
    if (vector instanceof Writable) {
      w = (Writable) vector;
    } else if(vector instanceof RandomAccessSparseVector) {
      w = new RandomAccessSparseVectorWritable(vector);
    } else if(vector instanceof SequentialAccessSparseVector) {
      w = new SequentialAccessSparseVectorWritable((SequentialAccessSparseVector)vector);
    } else {
      w = new DenseVectorWritable(new DenseVector(vector));
    }
    w.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    try {
      String vectorClassName = in.readUTF();
      Class<? extends Vector> vectorClass = Class.forName(vectorClassName).asSubclass(Vector.class);
      vector = vectorClass.newInstance();
      ((Writable)vector).readFields(in);
    } catch (ClassNotFoundException cnfe) {
      throw new IOException(cnfe);
    } catch (ClassCastException cce) {
      throw new IOException(cce);
    } catch (InstantiationException ie) {
      throw new IOException(ie);
    } catch (IllegalAccessException iae) {
      throw new IOException(iae);
    }
  }

  /** Write the vector to the output */
  public static void writeVector(DataOutput out, Vector vector) throws IOException {
    new VectorWritable(vector).write(out);
  }

}
