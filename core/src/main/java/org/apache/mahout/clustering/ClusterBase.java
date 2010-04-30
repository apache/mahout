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

package org.apache.mahout.clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.Locale;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

/**
 * ClusterBase is an abstract base class class for several clustering implementations
 * that share common implementations of various atttributes
 *
 */
public abstract class ClusterBase implements Writable, Cluster {

  // this cluster's clusterId
  private int id;

  // the current cluster center
  private Vector center;

  // the number of points in the cluster
  private int numPoints;

  // the Vector total of all points added to the cluster
  private Vector pointTotal;

  @Override
  public int getId() {
    return id;
  }

  public void setId(int id) {
    this.id = id;
  }

  @Override
  public Vector getCenter() {
    return center;
  }

  public void setCenter(Vector center) {
    this.center = center;
  }

  @Override
  public int getNumPoints() {
    return numPoints;
  }

  public void setNumPoints(int numPoints) {
    this.numPoints = numPoints;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  public void setPointTotal(Vector pointTotal) {
    this.pointTotal = pointTotal;
  }

  /**
   * @deprecated
   * @return
   */
  @Deprecated
  public abstract String asFormatString();

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder();
    buf.append(getIdentifier()).append(": ").append(formatVector(getCenter(), bindings));
    return buf.toString();
  }

  public abstract Vector computeCentroid();

  public abstract Object getIdentifier();

  @Override
  public String asJsonString() {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.toJson(this, this.getClass());
  }

  /**
   * Simply writes out the id, and that's it!
   * 
   * @param out
   *          The {@link java.io.DataOutput}
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
  }

  /** Reads in the id, nothing else */
  @Override
  public void readFields(DataInput in) throws IOException {
    id = in.readInt();
  }

  /**
   * Return a human-readable formatted string representation of the vector, not intended to be complete nor
   * usable as an input/output representation such as Json
   * 
   * @param v
   *          a Vector
   * @return a String
   */
  public static String formatVector(Vector v, String[] bindings) {
    StringBuilder buf = new StringBuilder();
    if (v instanceof NamedVector) {
      buf.append(((NamedVector) v).getName()).append(" = ");
    }
    int nzero = 0;
    Iterator<Element> iterateNonZero = v.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      iterateNonZero.next();
      nzero++;
    }
    // if vector is sparse or if we have bindings, use sparse notation
    if ((nzero < v.size()) || (bindings != null)) {
      buf.append('[');
      for (int i = 0; i < v.size(); i++) {
        double elem = v.get(i);
        if (elem == 0.0) {
          continue;
        }
        String label;
        if ((bindings != null) && ((label = bindings[i]) != null)) {
          buf.append(label).append(':');
        } else {
          buf.append(i).append(':');
        }
        buf.append(String.format(Locale.ENGLISH, "%.3f", elem)).append(", ");
      }
    } else {
      buf.append('[');
      for (int i = 0; i < v.size(); i++) {
        double elem = v.get(i);
        buf.append(String.format(Locale.ENGLISH, "%.3f", elem)).append(", ");
      }
    }
    if (buf.length() > 1) {
      buf.setLength(buf.length() - 2);
    }
    buf.append(']');
    return buf.toString();
  }
}
