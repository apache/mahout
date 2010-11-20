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

import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.SquareRootFunction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public abstract class AbstractCluster implements Cluster {

  private static final Type VECTOR_TYPE = new TypeToken<Vector>() {
  }.getType();

  // cluster persistent state
  private int id;

  private int numPoints;

  private Vector center;

  private Vector radius;

  protected AbstractCluster() {
  }

  protected AbstractCluster(Vector point, int id2) {
    this.numPoints = 0;
    this.center = new RandomAccessSparseVector(point);
    this.radius = point.like();
    this.id = id2;
  }

  protected AbstractCluster(Vector center2, Vector radius2, int id2) {
    this.numPoints = 0;
    this.center = new RandomAccessSparseVector(center2);
    this.radius = new RandomAccessSparseVector(radius2);
    this.id = id2;
  }

  /**
   * @param id the id to set
   */
  protected void setId(int id) {
    this.id = id;
  }

  /**
   * @param numPoints the numPoints to set
   */
  protected void setNumPoints(int numPoints) {
    this.numPoints = numPoints;
  }

  /**
   * @param center the center to set
   */
  protected void setCenter(Vector center) {
    this.center = center;
  }

  /**
   * @param radius the radius to set
   */
  protected void setRadius(Vector radius) {
    this.radius = radius;
  }

  // the observation statistics, initialized by the first observation
  private transient double s0;

  private transient Vector s1;

  private transient Vector s2;

  /**
   * @return the s0
   */
  protected double getS0() {
    return s0;
  }

  /**
   * @return the s1
   */
  protected Vector getS1() {
    return s1;
  }

  /**
   * @return the s2
   */
  protected Vector getS2() {
    return s2;
  }

  public void observe(ClusterObservations observations) {
    s0 += observations.getS0();
    if (s1 == null) {
      s1 = observations.getS1().clone();
    } else {
      observations.getS1().addTo(s1);
    }
    if (s2 == null) {
      s2 = observations.getS2().clone();
    } else {
      observations.getS2().addTo(s2);
    }
  }

  @Override
  public void observe(VectorWritable x) {
    observe(x.get());
  }

  public void observe(Vector x, double weight) {
    s0 += weight;
    Vector weightedX = x.times(weight);
    if (s1 == null) {
      s1 = weightedX;
    } else {
      weightedX.addTo(s1);
    }
    Vector x2 = x.times(x).times(weight);
    if (s2 == null) {
      s2 = x2;
    } else {
      x2.addTo(s2);
    }
  }

  public void observe(Vector x) {
    observe(x, 1.0);
  }

  @Override
  public int getNumPoints() {
    return numPoints;
  }

  public ClusterObservations getObservations() {
    return new ClusterObservations(s0, s1, s2);
  }

  @Override
  public void computeParameters() {
    if (s0 == 0) {
      return;
    }
    numPoints = (int) s0;
    center = s1.divide(s0);
    // compute the component stds
    if (s0 > 1) {
      radius = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
    } 
    s0 = 0;
    s1 = null;
    s2 = null;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readInt();
    this.numPoints = in.readInt();
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.center = temp.get();
    temp.readFields(in);
    this.radius = temp.get();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeInt(numPoints);
    VectorWritable.writeVector(out, center);
    VectorWritable.writeVector(out, radius);
  }

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder(50);
    buf.append(getIdentifier()).append("{n=").append(numPoints);
    if (center != null) {
      buf.append(" c=").append(formatVector(center, bindings));
    }
    if (radius != null) {
      buf.append(" r=").append(formatVector(radius, bindings));
    }
    buf.append('}');
    return buf.toString();
  }

  public abstract String getIdentifier();

  @Override
  public String asJsonString() {
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(VECTOR_TYPE, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.toJson(this, this.getClass());
  }

  @Override
  public Vector getCenter() {
    return center;
  }

  @Override
  public int getId() {
    return id;
  }

  @Override
  public Vector getRadius() {
    return radius;
  }

  /**
   * Compute the centroid by averaging the pointTotals
   * 
   * @return the new centroid
   */
  public Vector computeCentroid() {
    return s0 == 0 ? getCenter() : s1.divide(s0);
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
    Iterator<Vector.Element> iterateNonZero = v.iterateNonZero();
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

  @Override
  public int count() {
    return getNumPoints();
  }
}
