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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.graph.model.Vertex;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class VertexOrMarker implements Writable {

  private boolean marker;
  private Vertex connectingVertex;

  public static final VertexOrMarker MARKER = new VertexOrMarker(true);

  public VertexOrMarker() {}

  public VertexOrMarker(Vertex vertex) {
    connectingVertex = vertex;
    marker = false;
  }

  public VertexOrMarker(long vertexId) {
    this(new Vertex(vertexId));
  }

  private VertexOrMarker(boolean marker) {
    this.marker = true;
  }

  public boolean isMarker() {
    return marker;
  }

  public Vertex getVertex() {
    return connectingVertex;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(marker);
    if (!marker) {
      connectingVertex.write(out);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    connectingVertex = null;
    marker = in.readBoolean();
    if (!marker) {
      connectingVertex = new Vertex();
      connectingVertex.readFields(in);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof VertexOrMarker) {
      VertexOrMarker other = (VertexOrMarker) o;
      if (marker && other.marker) {
        return true;
      } else if (marker || other.marker) {
        return false;
      } else {
        return connectingVertex.equals(other.connectingVertex);
      }
    }
    return false;
  }

  @Override
  public int hashCode() {
    return 31 * (marker ? 1 : 0) + (connectingVertex != null ? connectingVertex.hashCode() : 0);
  }
}
