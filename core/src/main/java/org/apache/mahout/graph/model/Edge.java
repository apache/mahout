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

import com.google.common.collect.ComparisonChain;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/** A representation of a directed edge */
public class Edge implements WritableComparable<Edge>, Cloneable {

  private Vertex start;
  private Vertex end;

  public Edge() {}

  public Edge(Vertex start, Vertex end) {
    this.start = start;
    this.end = end;
  }

  public Edge(long startVertexID, long endVertexID) {
    this(new Vertex(startVertexID), new Vertex(endVertexID));
  }

  @Override
  public void write(DataOutput out) throws IOException {
    start.write(out);
    end.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    start = new Vertex();
    start.readFields(in);
    end = new Vertex();
    end.readFields(in);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof Edge) {
      Edge other = (Edge) o;
      return start.equals(other.start) && end.equals(other.end);
    }
    return false;
  }

  public Vertex startVertex() {
    return start;
  }

  public Vertex endVertex() {
    return end;
  }

  @Override
  public int hashCode() {
    return start.hashCode() + 31 * end.hashCode();
  }

  @Override
  public String toString() {
    return "(" + start.getId() + ',' + end.getId() + ')';
  }

  @Override
  public int compareTo(Edge other) {
    return ComparisonChain.start()
        .compare(start, other.start)
        .compare(end, other.end).result();
  }

  @Override
  public Edge clone() {
    return new Edge(start.clone(), end.clone());
  }
}
