/**
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license
 * agreements. See the NOTICE file distributed with this work for additional information regarding
 * copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.apache.mahout.math;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector.Element;

import com.google.common.base.Preconditions;

public final class VectorWritable extends Configured implements Writable {

  public static final int FLAG_DENSE = 0x01;
  public static final int FLAG_SEQUENTIAL = 0x02;
  public static final int FLAG_NAMED = 0x04;
  public static final int FLAG_LAX_PRECISION = 0x08;
  public static final int NUM_FLAGS = 4;

  private Vector vector;
  private boolean writesLaxPrecision;

  public VectorWritable() {}

  public VectorWritable(boolean writesLaxPrecision) {
    setWritesLaxPrecision(writesLaxPrecision);
  }

  public VectorWritable(Vector vector) {
    this.vector = vector;
  }

  public VectorWritable(Vector vector, boolean writesLaxPrecision) {
    this(vector);
    setWritesLaxPrecision(writesLaxPrecision);
  }

  /**
   * @return {@link Vector} that this is to write, or has
   *  just read
   */
  public Vector get() {
    return vector;
  }

  public void set(Vector vector) {
    this.vector = vector;
  }

  /**
   * @return true if this is allowed to encode {@link Vector}
   *  values using fewer bytes, possibly losing precision. In particular this means
   *  that floating point values will be encoded as floats, not doubles.
   */
  public boolean isWritesLaxPrecision() {
    return writesLaxPrecision;
  }

  public void setWritesLaxPrecision(boolean writesLaxPrecision) {
    this.writesLaxPrecision = writesLaxPrecision;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    writeVector(out, this.vector, this.writesLaxPrecision);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int flags = in.readByte();
    Preconditions.checkArgument(flags >> NUM_FLAGS == 0, "Unknown flags set: %d", Integer.toString(flags, 2));
    boolean dense = (flags & FLAG_DENSE) != 0;
    boolean sequential = (flags & FLAG_SEQUENTIAL) != 0;
    boolean named = (flags & FLAG_NAMED) != 0;
    boolean laxPrecision = (flags & FLAG_LAX_PRECISION) != 0;

    int size = Varint.readUnsignedVarInt(in);
    Vector v;
    if (dense) {
      double[] values = new double[size];
      for (int i = 0; i < size; i++) {
        values[i] = laxPrecision ? in.readFloat() : in.readDouble();
      }
      v = new DenseVector(values);
    } else {
      int numNonDefaultElements = Varint.readUnsignedVarInt(in);
      v = sequential
          ? new SequentialAccessSparseVector(size, numNonDefaultElements)
          : new RandomAccessSparseVector(size, numNonDefaultElements);
      if (sequential) {
        int lastIndex = 0;
        for (int i = 0; i < numNonDefaultElements; i++) {
          int delta = Varint.readUnsignedVarInt(in);
          int index = lastIndex + delta;
          lastIndex = index;
          double value = laxPrecision ? in.readFloat() : in.readDouble();
          v.setQuick(index, value);
        }
      } else {
        for (int i = 0; i < numNonDefaultElements; i++) {
          int index = Varint.readUnsignedVarInt(in);
          double value = laxPrecision ? in.readFloat() : in.readDouble();
          v.setQuick(index, value);
        }
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
    writeVector(out, vector, false);
  }

  public static void writeVector(DataOutput out, Vector vector, boolean laxPrecision) throws IOException {
    boolean dense = vector.isDense();
    boolean sequential = vector.isSequentialAccess();
    boolean named = vector instanceof NamedVector;

    out.writeByte((dense ? FLAG_DENSE : 0)
        | (sequential ? FLAG_SEQUENTIAL : 0)
        | (named ? FLAG_NAMED : 0)
        | (laxPrecision ? FLAG_LAX_PRECISION : 0));

    Varint.writeUnsignedVarInt(vector.size(), out);
    if (dense) {
      for (Vector.Element element : vector.all()) {
        if (laxPrecision) {
          out.writeFloat((float) element.get());
        } else {
          out.writeDouble(element.get());
        }
      }
    } else {
      Varint.writeUnsignedVarInt(vector.getNumNonZeroElements(), out);
      Iterator<Element> iter = vector.nonZeroes().iterator();
      if (sequential) {
        int lastIndex = 0;
        while (iter.hasNext()) {
          Vector.Element element = iter.next();
          if (element.get() == 0) {
            continue;
          }
          int thisIndex = element.index();
          // Delta-code indices:
          Varint.writeUnsignedVarInt(thisIndex - lastIndex, out);
          lastIndex = thisIndex;
          if (laxPrecision) {
            out.writeFloat((float) element.get());
          } else {
            out.writeDouble(element.get());
          }
        }
      } else {
        while (iter.hasNext()) {
          Vector.Element element = iter.next();
          if (element.get() == 0) {
            // TODO(robinanil): Fix the damn iterator for the zero element.
            continue;
          }
          Varint.writeUnsignedVarInt(element.index(), out);
          if (laxPrecision) {
            out.writeFloat((float) element.get());
          } else {
            out.writeDouble(element.get());
          }
        }
      }
    }
    if (named) {
      String name = ((NamedVector) vector).getName();
      out.writeUTF(name == null ? "" : name);
    }
  }

  public static Vector readVector(DataInput in) throws IOException {
    VectorWritable v = new VectorWritable();
    v.readFields(in);
    return v.get();
  }

  public static VectorWritable merge(Iterator<VectorWritable> vectors) {
    return new VectorWritable(mergeToVector(vectors));
  }

  public static Vector mergeToVector(Iterator<VectorWritable> vectors) {
    Vector accumulator = vectors.next().get();
    while (vectors.hasNext()) {
      VectorWritable v = vectors.next();
      if (v != null) {
        for (Element nonZeroElement : v.get().nonZeroes()) {
          accumulator.setQuick(nonZeroElement.index(), nonZeroElement.get());
        }
      }
    }
    return accumulator;
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof VectorWritable && vector.equals(((VectorWritable) o).get());
  }

  @Override
  public int hashCode() {
    return vector.hashCode();
  }

  @Override
  public String toString() {
    return vector.toString();
  }
}
