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
import java.lang.reflect.InvocationTargetException;

public class VectorWritable extends Configured implements Writable {

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
    VectorWritable writable;
    Class<? extends Vector> vectorClass = vector.getClass();
    String writableClassName = vectorClass.getName() + "Writable";
    try {
      Class<? extends VectorWritable> vectorWritableClass =
          Class.forName(writableClassName).asSubclass(VectorWritable.class);
      writable = vectorWritableClass.getConstructor(vectorClass).newInstance(vector);
    } catch (ClassNotFoundException cnfe) {
      throw new IOException(cnfe);
    } catch (NoSuchMethodException nsme) {
      throw new IOException(nsme);
    } catch (InvocationTargetException ite) {
      throw new IOException(ite);
    } catch (InstantiationException ie) {
      throw new IOException(ie);
    } catch (IllegalAccessException iae) {
      throw new IOException(iae);
    }
    out.writeUTF(writableClassName);
    writable.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    String writableClassName = in.readUTF();
    try {
      Class<? extends VectorWritable> writableClass =
          Class.forName(writableClassName).asSubclass(VectorWritable.class);
      VectorWritable writable = writableClass.getConstructor().newInstance();
      writable.readFields(in);
      vector = writable.get();
    } catch (ClassNotFoundException cnfe) {
      throw new IOException(cnfe);
    } catch (ClassCastException cce) {
      throw new IOException(cce);
    } catch (InstantiationException ie) {
      throw new IOException(ie);
    } catch (IllegalAccessException iae) {
      throw new IOException(iae);
    } catch (NoSuchMethodException nsme) {
      throw new IOException(nsme);
    } catch (InvocationTargetException ite) {
      throw new IOException(ite);
    }
  }

  /** Write the vector to the output */
  public static void writeVector(DataOutput out, Vector vector) throws IOException {
    new VectorWritable(vector).write(out);
  }

}
