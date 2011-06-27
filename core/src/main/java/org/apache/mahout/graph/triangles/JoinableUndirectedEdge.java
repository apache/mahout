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

package org.apache.mahout.graph.triangles;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.Vertex;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

public class JoinableUndirectedEdge implements WritableComparable<JoinableUndirectedEdge> {

  private UndirectedEdge edge;
  private boolean marked;

  static {
    WritableComparator.define(JoinableUndirectedEdge.class, new SecondarySortComparator());
  }

  public JoinableUndirectedEdge() {}

  public JoinableUndirectedEdge(UndirectedEdge edge, boolean marked) {
    this.edge = edge;
    this.marked = marked;
  }

  public JoinableUndirectedEdge(Vertex firstVertex, Vertex secondVertex, boolean marked) {
    this(new UndirectedEdge(firstVertex, secondVertex), marked);
  }

  public JoinableUndirectedEdge(long firstVertexId, long secondVertexId, boolean marked) {
    this(new UndirectedEdge(firstVertexId, secondVertexId), marked);
  }

  public UndirectedEdge getEdge() {
    return edge;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    edge.write(out);
    out.writeBoolean(marked);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    edge = new UndirectedEdge();
    edge.readFields(in);
    marked = in.readBoolean();
  }

  @Override
  public int compareTo(JoinableUndirectedEdge other) {
    return edge.compareTo(other.edge);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof JoinableUndirectedEdge) {
      return edge.equals(((JoinableUndirectedEdge) o).edge);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return edge.hashCode();
  }

  @Override
  public String toString() {
    return "(" + edge + ',' + marked + ')';
  }

  public static class SecondarySortComparator extends WritableComparator implements Serializable {

    protected SecondarySortComparator() {
      super(JoinableUndirectedEdge.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      JoinableUndirectedEdge first = (JoinableUndirectedEdge) a;
      JoinableUndirectedEdge second = (JoinableUndirectedEdge) b;

      int result = first.edge.compareTo(second.edge);
      if (result == 0) {
        if (first.marked && !second.marked) {
          return -1;
        } else if (!first.marked && second.marked) {
          return 1;
        }
      }
      return result;
    }

  }

  public static class GroupingComparator extends WritableComparator implements Serializable {
    protected GroupingComparator() {
      super(JoinableUndirectedEdge.class, true);
    }
  }
}
