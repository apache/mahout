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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class VertexWithDegree implements Writable, Cloneable {

  private Vertex vertex;
  private int degree;

  public VertexWithDegree() {}

  public VertexWithDegree(Vertex vertex, int degree) {
    this.vertex = vertex;
    this.degree = degree;
  }

  public VertexWithDegree(long vertexId, int degree) {
    this(new Vertex(vertexId), degree);
  }

  public Vertex getVertex() {
    return vertex;
  }

  public int getDegree() {
    return degree;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    vertex.write(out);
    Varint.writeUnsignedVarInt(degree, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    vertex = new Vertex();
    vertex.readFields(in);
    degree = Varint.readUnsignedVarInt(in);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof VertexWithDegree) {
      return vertex.equals(((VertexWithDegree) o).vertex);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return vertex.hashCode();
  }

  @Override
  public VertexWithDegree clone() {
    return new VertexWithDegree(vertex.clone(), degree);
  }

  @Override
  public String toString() {
    return "(" + vertex + ',' + degree + ')';
  }
}
