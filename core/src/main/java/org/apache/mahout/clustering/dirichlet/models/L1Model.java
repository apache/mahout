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

import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.dirichlet.JsonModelAdapter;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class L1Model implements Model<VectorWritable> {

  private static final DistanceMeasure measure = new ManhattanDistanceMeasure();

  private int id;

  public L1Model() {
    super();
  }

  public L1Model(int id, Vector v) {
    this.id = id;
    observed = v.like();
    coefficients = v;
  }

  private Vector coefficients;

  private int count = 0;

  private Vector observed;

  private static final Type modelType = new TypeToken<Model<Vector>>() {
  }.getType();

  @Override
  public void computeParameters() {
    coefficients = observed.divide(count);
  }

  @Override
  public int count() {
    return count;
  }

  @Override
  public void observe(VectorWritable x) {
    count++;
    x.get().addTo(observed);
  }

  @Override
  public double pdf(VectorWritable x) {
    return Math.exp(-L1Model.measure.distance(x.get(), coefficients));
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.id = in.readInt();
    this.count = in.readInt();
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.coefficients = temp.get();
    this.observed = coefficients.like();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeInt(count);
    VectorWritable.writeVector(out, coefficients);
  }

  public L1Model sample() {
    return new L1Model(id, coefficients.clone());
  }

  @Override
  public String toString() {
    return asFormatString(null);
  }

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder();
    buf.append("l1m{n=").append(count).append(" c=");
    if (coefficients != null) {
      buf.append(ClusterBase.formatVector(coefficients, bindings));
    }
    buf.append('}');
    return buf.toString();
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.clustering.Printable#asJsonString()
   */
  @Override
  public String asJsonString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, modelType);
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
    return count;
  }

}
