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

package org.apache.mahout.clustering.dirichlet.models;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.JsonModelAdapter;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.UncommonDistributions;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.SquareRootFunction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class AsymmetricSampledNormalModel implements Cluster {

  private static final Type MODEL_TYPE = new TypeToken<Model<Vector>>() {
  }.getType();

  private int id;

  // the parameters
  private Vector mean;

  private Vector stdDev;

  // the observation statistics, initialized by the first observation
  private int s0;

  private Vector s1;

  private Vector s2;

  public AsymmetricSampledNormalModel() {
  }

  public AsymmetricSampledNormalModel(int id, Vector mean, Vector stdDev) {
    this.id = id;
    this.mean = mean;
    this.stdDev = stdDev;
    this.s0 = 0;
    this.s1 = mean.like();
    this.s2 = mean.like();
  }

  public Vector getMean() {
    return mean;
  }

  public Vector getStdDev() {
    return stdDev;
  }

  /**
   * Return an instance with the same parameters
   * 
   * @return an AsymmetricSampledNormalModel
   */
  @Override
  public AsymmetricSampledNormalModel sampleFromPosterior() {
    return new AsymmetricSampledNormalModel(id, mean, stdDev);
  }

  @Override
  public void observe(VectorWritable v) {
    Vector x = v.get();
    s0++;
    if (s1 == null) {
      s1 = x.clone();
    } else {
      s1 = s1.plus(x);
    }
    if (s2 == null) {
      s2 = x.times(x);
    } else {
      s2 = s2.plus(x.times(x));
    }
  }

  @Override
  public void computeParameters() {
    if (s0 == 0) {
      return;
    }
    mean = s1.divide(s0);
    // compute the component stds
    if (s0 > 1) {
      stdDev = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
    } else {
      stdDev.assign(Double.MIN_NORMAL);
    }
  }

  @Override
  public double pdf(VectorWritable v) {
    Vector x = v.get();
    // return the product of the component pdfs
    // TODO: is this reasonable? correct? It seems to work in some cases.
    double pdf = 1;
    for (int i = 0; i < x.size(); i++) {
      // small prior on stdDev to avoid numeric instability when stdDev==0
      pdf *= UncommonDistributions.dNorm(x.getQuick(i), mean.getQuick(i), stdDev.getQuick(i) + 0.000001);
    }
    return pdf;
  }

  @Override
  public int count() {
    return s0;
  }

  @Override
  public String toString() {
    return asFormatString(null);
  }

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder(50);
    buf.append("asnm{n=").append(s0).append(" m=");
    if (mean != null) {
      buf.append(AbstractCluster.formatVector(mean, bindings));
    }
    buf.append(" sd=");
    if (stdDev != null) {
      buf.append(AbstractCluster.formatVector(stdDev, bindings));
    }
    buf.append('}');
    return buf.toString();
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readInt();
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.mean = temp.get();
    temp.readFields(in);
    this.stdDev = temp.get();
    this.s0 = in.readInt();
    temp.readFields(in);
    this.s1 = temp.get();
    temp.readFields(in);
    this.s2 = temp.get();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    VectorWritable.writeVector(out, mean);
    VectorWritable.writeVector(out, stdDev);
    out.writeInt(s0);
    VectorWritable.writeVector(out, s1);
    VectorWritable.writeVector(out, s2);
  }

  @Override
  public String asJsonString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, MODEL_TYPE);
  }

  @Override
  public Vector getCenter() {
    return mean;
  }

  @Override
  public int getId() {
    return id;
  }

  @Override
  public int getNumPoints() {
    return s0;
  }

  @Override
  public Vector getRadius() {
    return getStdDev();
  }
}
