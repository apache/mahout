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

public class UndirectedEdgeWithDegrees implements WritableComparable<UndirectedEdgeWithDegrees> {

  private VertexWithDegree first;
  private VertexWithDegree second;

  public UndirectedEdgeWithDegrees() {}

  public UndirectedEdgeWithDegrees(VertexWithDegree first, VertexWithDegree second) {
    if (first.getVertex().compareTo(second.getVertex()) < 0) {
      this.first = first;
      this.second = second;
    } else {
      this.first = second;
      this.second = first;
    }
  }

  public UndirectedEdgeWithDegrees(long firstVertexId, int firstVertexDegree, long secondVertexId,
      int secondVertexDegree) {
    this(new VertexWithDegree(firstVertexId, firstVertexDegree), new VertexWithDegree(secondVertexId,
        secondVertexDegree));
  }

  @Override
  public void write(DataOutput out) throws IOException {
    first.write(out);
    second.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    first = new VertexWithDegree();
    first.readFields(in);
    second = new VertexWithDegree();
    second.readFields(in);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof UndirectedEdgeWithDegrees) {
      UndirectedEdgeWithDegrees other = (UndirectedEdgeWithDegrees) o;
      if (first.equals(other.first) && second.equals(other.second)) {
        return true;
      }
    }
    return false;
  }

  public VertexWithDegree getFirstVertexWithDegree() {
    return first;
  }

  public VertexWithDegree getSecondVertexWithDegree() {
    return second;
  }

  @Override
  public int hashCode() {
    return first.hashCode() + 31 * second.hashCode();
  }

  @Override
  public String toString() {
    return "(" + first + ", " + second + ')';
  }

  @Override
  public int compareTo(UndirectedEdgeWithDegrees other) {
    return ComparisonChain.start()
        .compare(first.getVertex(), other.first.getVertex())
        .compare(second.getVertex(), other.second.getVertex()).result();
  }
}
