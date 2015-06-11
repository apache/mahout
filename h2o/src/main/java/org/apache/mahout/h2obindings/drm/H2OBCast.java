/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings.drm;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.drm.BCast;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Broadcast class wrapper around Matrix and Vector.
 *
 * Use MatrixWritable and VectorWritable internally.
 * Even though the class is generically typed, we do runtime
 * enforcement to assert the type is either Matrix or Vector.
 *
 * H2OBCast object is created around a Matrix or Vector. Matrix or Vector
 * objects cannot be freely referred in closures. Instead create and refer the
 * corresponding H2OBCast object. The original Matrix or Vector can be
 * obtained by calling the ->value() method on the H2OBCast object within a
 * closure.
 */
public class H2OBCast<T> implements BCast<T>, Serializable {
  private transient T obj;
  private byte buf[];
  private boolean isMatrix;

  /**
   * Class constructor.
   */
  public H2OBCast(T o) {
    obj = o;
    if (o instanceof Matrix) {
      buf = serialize(new MatrixWritable((Matrix)o));
      isMatrix = true;
    } else if (o instanceof Vector) {
      buf = serialize(new VectorWritable((Vector)o));
      isMatrix = false;
    } else {
      throw new IllegalArgumentException("Only Matrix or Vector supported for now");
    }
  }

  /**
   * Get the serialized object.
   */
  public T value() {
    if (obj == null) {
      obj = deserialize(buf);
    }
    return obj;
  }

  /**
   * Internal method to serialize the object.
   *
   * @param w Either MatrixWritable or VectorWritable corresponding to
   *          either Matrix or Vector as the class is typed.
   * @return serialized sequence of bytes.
   */
  private byte[] serialize(Writable w) {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    try {
      ObjectOutputStream oos = new ObjectOutputStream(bos);
      w.write(oos);
      oos.close();
    } catch (IOException e) {
      return null;
    }
    return bos.toByteArray();
  }

  /**
   * Internal method to deserialize a sequence of bytes.
   *
   * @param buf Sequence of bytes previously serialized by serialize() method.
   * @return The original Matrix or Vector object.
   */
  private T deserialize(byte buf[]) {
    T ret = null;
    try (ByteArrayInputStream bis = new ByteArrayInputStream(buf)){
      ObjectInputStream ois = new ObjectInputStream(bis);
      if (isMatrix) {
        MatrixWritable w = new MatrixWritable();
        w.readFields(ois);
        ret = (T) w.get();
      } else {
        VectorWritable w = new VectorWritable();
        w.readFields(ois);
        ret = (T) w.get();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
    return ret;
  }

  /**
   * Stop broadcasting when called on driver side. Release any network resources.
   *
   */
  @Override
  public void close() throws IOException {

    // TODO: review this. It looks like it is not really a broadcast mechanism but rather just a
    // serialization wrapper. In which case it doesn't hold any network resources.

  }
}
