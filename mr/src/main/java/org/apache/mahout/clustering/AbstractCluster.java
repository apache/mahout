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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.SquareRootFunction;
import org.codehaus.jackson.map.ObjectMapper;

public abstract class AbstractCluster implements Cluster {
  
  // cluster persistent state
  private int id;
  
  private long numObservations;
  
  private long totalObservations;
  
  private Vector center;
  
  private Vector radius;
  
  // the observation statistics
  private double s0;
  
  private Vector s1;
  
  private Vector s2;

  private static final ObjectMapper jxn = new ObjectMapper();
  
  protected AbstractCluster() {}
  
  protected AbstractCluster(Vector point, int id2) {
    this.numObservations = (long) 0;
    this.totalObservations = (long) 0;
    this.center = point.clone();
    this.radius = center.like();
    this.s0 = (double) 0;
    this.s1 = center.like();
    this.s2 = center.like();
    this.id = id2;
  }
  
  protected AbstractCluster(Vector center2, Vector radius2, int id2) {
    this.numObservations = (long) 0;
    this.totalObservations = (long) 0;
    this.center = new RandomAccessSparseVector(center2);
    this.radius = new RandomAccessSparseVector(radius2);
    this.s0 = (double) 0;
    this.s1 = center.like();
    this.s2 = center.like();
    this.id = id2;
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeLong(getNumObservations());
    out.writeLong(getTotalObservations());
    VectorWritable.writeVector(out, getCenter());
    VectorWritable.writeVector(out, getRadius());
    out.writeDouble(s0);
    VectorWritable.writeVector(out, s1);
    VectorWritable.writeVector(out, s2);
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readInt();
    this.setNumObservations(in.readLong());
    this.setTotalObservations(in.readLong());
    this.setCenter(VectorWritable.readVector(in));
    this.setRadius(VectorWritable.readVector(in));
    this.setS0(in.readDouble());
    this.setS1(VectorWritable.readVector(in));
    this.setS2(VectorWritable.readVector(in));
  }
  
  @Override
  public void configure(Configuration job) {
    // nothing to do
  }
  
  @Override
  public Collection<Parameter<?>> getParameters() {
    return Collections.emptyList();
  }
  
  @Override
  public void createParameters(String prefix, Configuration jobConf) {
    // nothing to do
  }
  
  @Override
  public int getId() {
    return id;
  }

  /**
   * @param id
   *          the id to set
   */
  protected void setId(int id) {
    this.id = id;
  }
  
  @Override
  public long getNumObservations() {
    return numObservations;
  }

  /**
   * @param l
   *          the numPoints to set
   */
  protected void setNumObservations(long l) {
    this.numObservations = l;
  }
  
  @Override
  public long getTotalObservations() {
    return totalObservations;
  }

  protected void setTotalObservations(long totalPoints) {
    this.totalObservations = totalPoints;
  }

  @Override
  public Vector getCenter() {
    return center;
  }

  /**
   * @param center
   *          the center to set
   */
  protected void setCenter(Vector center) {
    this.center = center;
  }
  
  @Override
  public Vector getRadius() {
    return radius;
  }

  /**
   * @param radius
   *          the radius to set
   */
  protected void setRadius(Vector radius) {
    this.radius = radius;
  }
  
  /**
   * @return the s0
   */
  protected double getS0() {
    return s0;
  }
  
  protected void setS0(double s0) {
    this.s0 = s0;
  }

  /**
   * @return the s1
   */
  protected Vector getS1() {
    return s1;
  }
  
  protected void setS1(Vector s1) {
    this.s1 = s1;
  }

  /**
   * @return the s2
   */
  protected Vector getS2() {
    return s2;
  }
  
  protected void setS2(Vector s2) {
    this.s2 = s2;
  }

  @Override
  public void observe(Model<VectorWritable> x) {
    AbstractCluster cl = (AbstractCluster) x;
    setS0(getS0() + cl.getS0());
    setS1(getS1().plus(cl.getS1()));
    setS2(getS2().plus(cl.getS2()));
  }
  
  @Override
  public void observe(VectorWritable x) {
    observe(x.get());
  }
  
  @Override
  public void observe(VectorWritable x, double weight) {
    observe(x.get(), weight);
  }
  
  public void observe(Vector x, double weight) {
    if (weight == 1.0) {
      observe(x);
    } else {
      setS0(getS0() + weight);
      Vector weightedX = x.times(weight);
      if (getS1() == null) {
        setS1(weightedX);
      } else {
        getS1().assign(weightedX, Functions.PLUS);
      }
      Vector x2 = x.times(x).times(weight);
      if (getS2() == null) {
        setS2(x2);
      } else {
        getS2().assign(x2, Functions.PLUS);
      }
    }
  }
  
  public void observe(Vector x) {
    setS0(getS0() + 1);
    if (getS1() == null) {
      setS1(x.clone());
    } else {
      getS1().assign(x, Functions.PLUS);
    }
    Vector x2 = x.times(x);
    if (getS2() == null) {
      setS2(x2);
    } else {
      getS2().assign(x2, Functions.PLUS);
    }
  }
  
  
  @Override
  public void computeParameters() {
    if (getS0() == 0) {
      return;
    }
    setNumObservations((long) getS0());
    setTotalObservations(getTotalObservations() + getNumObservations());
    setCenter(getS1().divide(getS0()));
    // compute the component stds
    if (getS0() > 1) {
      setRadius(getS2().times(getS0()).minus(getS1().times(getS1())).assign(new SquareRootFunction()).divide(getS0()));
    }
    setS0(0);
    setS1(center.like());
    setS2(center.like());
  }

  @Override
  public String asFormatString(String[] bindings) {
    String fmtString = "";
    try {
      fmtString = jxn.writeValueAsString(asJson(bindings));
    } catch (IOException e) {
      log.error("Error writing JSON as String.", e);
    }
    return fmtString;
  }

  public Map<String,Object> asJson(String[] bindings) {
    Map<String,Object> dict = new HashMap<>();
    dict.put("identifier", getIdentifier());
    dict.put("n", getNumObservations());
    if (getCenter() != null) {
      try {
        dict.put("c", formatVectorAsJson(getCenter(), bindings));
      } catch (IOException e) {
        log.error("IOException:  ", e);
      }
    }
    if (getRadius() != null) {
      try {
        dict.put("r", formatVectorAsJson(getRadius(), bindings));
      } catch (IOException e) {
        log.error("IOException:  ", e);
      }
    }
    return dict;
  }
  
  public abstract String getIdentifier();
  
  /**
   * Compute the centroid by averaging the pointTotals
   * 
   * @return the new centroid
   */
  public Vector computeCentroid() {
    return getS0() == 0 ? getCenter() : getS1().divide(getS0());
  }

  /**
   * Return a human-readable formatted string representation of the vector, not
   * intended to be complete nor usable as an input/output representation
   */
  public static String formatVector(Vector v, String[] bindings) {
    String fmtString = "";
    try {
      fmtString = jxn.writeValueAsString(formatVectorAsJson(v, bindings));
    } catch (IOException e) {
      log.error("Error writing JSON as String.", e);
    }
    return fmtString;
  }

  /**
   * Create a List of HashMaps containing vector terms and weights
   *
   * @return List<Object>
   */
  public static List<Object> formatVectorAsJson(Vector v, String[] bindings) throws IOException {

    boolean hasBindings = bindings != null;
    boolean isSparse = v.getNumNonZeroElements() != v.size();

    // we assume sequential access in the output
    Vector provider = v.isSequentialAccess() ? v : new SequentialAccessSparseVector(v);

    List<Object> terms = new LinkedList<>();
    String term = "";

    for (Element elem : provider.nonZeroes()) {

      if (hasBindings && bindings.length >= elem.index() + 1 && bindings[elem.index()] != null) {
        term = bindings[elem.index()];
      } else if (hasBindings || isSparse) {
        term = String.valueOf(elem.index());
      }

      Map<String, Object> term_entry = new HashMap<>();
      double roundedWeight = (double) Math.round(elem.get() * 1000) / 1000;
      if (hasBindings || isSparse) {
        term_entry.put(term, roundedWeight);
        terms.add(term_entry);
      } else {
        terms.add(roundedWeight);
      }
    }

    return terms;
  }

  @Override
  public boolean isConverged() {
    // Convergence has no meaning yet, perhaps in subclasses
    return false;
  }
}
