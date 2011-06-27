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

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class Triangle implements Writable {

  private Vertex first;
  private Vertex second;
  private Vertex third;

  public Triangle() {}

  public Triangle(Vertex first, Vertex second, Vertex third) {
    List<Vertex> vertices = Lists.newArrayList(first, second, third);
    Collections.sort(vertices);
    this.first = vertices.get(0);
    this.second = vertices.get(1);
    this.third = vertices.get(2);
  }

  public Triangle(long firstVertexId, long secondVertexId, long thirdVertexId) {
    this(new Vertex(firstVertexId), new Vertex(secondVertexId), new Vertex(thirdVertexId));
  }

  public Vertex getFirstVertex() {
    return first;
  }

  public Vertex getSecondVertex() {
    return second;
  }

  public Vertex getThirdVertex() {
    return third;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    first.write(out);
    second.write(out);
    third.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    first = new Vertex();
    first.readFields(in);
    second = new Vertex();
    second.readFields(in);
    third = new Vertex();
    third.readFields(in);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof Triangle) {
      Triangle other = (Triangle) o;
      return first.equals(other.first) && second.equals(other.second) && third.equals(other.third);
    }
    return false;
  }

  @Override
  public int hashCode() {
    int result = 31 * first.hashCode() + second.hashCode();
    return 31 * result + third.hashCode();
  }

  @Override
  public String toString() {
    return "(" + first.getId() + ',' + second.getId() + ',' + third.getId() + ')';
  }
}
