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

package org.apache.mahout.graph.model;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.google.common.primitives.Longs;
import org.apache.hadoop.io.WritableComparable;

/** models a vertex in a graph */
public class Vertex implements WritableComparable<Vertex>, Cloneable {

  private long id;

  public Vertex() {}

  public Vertex(long id) {
    this.id = id;
  }

  public static Vertex read(DataInput in) throws IOException {
    Vertex v = new Vertex();
    v.readFields(in);
    return v;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readLong();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(id);
  }

  public long getId() {
    return this.id;
  }

  /** Compares this instance to another according to the {@code id} attribute. */
  @Override
  public int compareTo(Vertex other) {
    return Longs.compare(id, other.id);
  }

  /** Compares this instance to another according to the {@code id} attribute */
  @Override
  public boolean equals(Object other) {
    if (other instanceof Vertex) {
      return ((Vertex) other).id == id;
    }
    return false;
  }

  @Override
  public Vertex clone() {
    return new Vertex(id);
  }

  /**
   * The hash code the {@code id} attribute
   */
  @Override
  public int hashCode() {
    return Longs.hashCode(id);
  }
  
  @Override
  public String toString() {
    return "(" + id + ')';
  }

}
