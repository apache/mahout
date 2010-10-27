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
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class L1Model implements Cluster {

  private static final DistanceMeasure MEASURE = new ManhattanDistanceMeasure();
  private static final Type MODEL_TYPE = new TypeToken<Model<Vector>>() {}.getType();

  private int id;

  private Vector coefficients;

  private int counter;

  private Vector observed;

  public L1Model() {
  }

  public L1Model(int id, Vector v) {
    this.id = id;
    observed = v.like();
    coefficients = v;
  }

  @Override
  public void computeParameters() {
    coefficients = observed.divide(counter);
  }

  @Override
  public int count() {
    return counter;
  }

  @Override
  public void observe(VectorWritable x) {
    counter++;
    x.get().addTo(observed);
  }

  @Override
  public double pdf(VectorWritable x) {
    return Math.exp(-L1Model.MEASURE.distance(x.get(), coefficients));
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readInt();
    this.counter = in.readInt();
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.coefficients = temp.get();
    this.observed = coefficients.like();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeInt(counter);
    VectorWritable.writeVector(out, coefficients);
  }

  @Override
  public L1Model sampleFromPosterior() {
    return new L1Model(id, coefficients.clone());
  }

  @Override
  public String toString() {
    return asFormatString(null);
  }

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder();
    buf.append("l1m{n=").append(counter).append(" c=");
    if (coefficients != null) {
      buf.append(AbstractCluster.formatVector(coefficients, bindings));
    }
    buf.append('}');
    return buf.toString();
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
    return coefficients;
  }

  @Override
  public int getId() {
    return id;
  }

  @Override
  public int getNumPoints() {
    return counter;
  }

  @Override
  public Vector getRadius() {
    return null;
  }

}
