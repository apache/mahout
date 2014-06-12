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

import org.apache.mahout.math.drm.BCast;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.VectorWritable;
import java.io.Serializable;
import java.io.ByteArrayOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;

/* Handle Matrix and Vector separately so that we can live with
   just importing MatrixWritable and VectorWritable.

   We could collapse the two into a single method using Writable,
   but then we would have to import org.apache.hadoop.Writable,
   pick a hadoop distribution in pom.xml etc. Instead let
   mahout-mrlegacy solve that transitively for us.
*/

public class H2OBCast<T> implements BCast<T>, Serializable {
  transient T obj;
  byte buf[];
  boolean is_matrix;

  public H2OBCast(T o) {
    obj = o;

    if (o instanceof Matrix) {
      buf = serializeMatrix((Matrix)o);
      is_matrix = true;
    } else if (o instanceof Vector) {
      buf = serializeVector((Vector)o);
    } else {
      throw new IllegalArgumentException("Only Matrix or Vector supported for now");
    }
  }

  public T value() {
    if (obj == null)
      obj = deserialize(buf);
    return obj;
  }

  private byte[] serializeMatrix(Matrix m) {
    MatrixWritable w = new MatrixWritable(m);
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    try {
      ObjectOutputStream oos = new ObjectOutputStream(bos);
      w.write(oos);
      oos.close();
    } catch (java.io.IOException e) {
      return null;
    }
    return bos.toByteArray();
  }

  private byte[] serializeVector(Vector v) {
    VectorWritable w = new VectorWritable(v);
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    try {
      ObjectOutputStream oos = new ObjectOutputStream(bos);
      w.write(oos);
      oos.close();
    } catch (java.io.IOException e) {
      return null;
    }
    return bos.toByteArray();
  }

  private T deserialize(byte buf[]) {
    T ret = null;
    ByteArrayInputStream bis = new ByteArrayInputStream(buf);
    try {
      ObjectInputStream ois = new ObjectInputStream(bis);
      if (is_matrix) {
        MatrixWritable w = new MatrixWritable();
        w.readFields(ois);
        ret = (T) w.get();
      } else {
        VectorWritable w = new VectorWritable();
        w.readFields(ois);
        ret = (T) w.get();
      }
    } catch (java.io.IOException e) {
      System.out.println("Caught exception: " + e);
    }
    return ret;
  }
}
