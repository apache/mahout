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
package org.apache.mahout.clustering.dirichlet;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class DirichletCluster implements Cluster {

  private static final Type CLUSTER_TYPE = new TypeToken<DirichletCluster>() {}.getType();

  private Cluster model; // the model for this iteration

  private double totalCount; // total count of observations for the model

  public DirichletCluster(Cluster model, double totalCount) {
    this.model = model;
    this.totalCount = totalCount;
  }

  public DirichletCluster(Cluster model) {
    this.model = model;
    this.totalCount = 0.0;
  }

  public DirichletCluster() {
  }

  public Cluster getModel() {
    return model;
  }

  public void setModel(Cluster model) {
    this.model = model;
    this.totalCount += model.count();
  }

  public double getTotalCount() {
    return totalCount;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.totalCount = in.readDouble();
    this.model = readModel(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(totalCount);
    writeModel(out, model);
  }

  /************* Methods required by Cluster *****************/
  
  /** Writes a typed Model instance to the output stream */
  public static void writeModel(DataOutput out, Model<?> model) throws IOException {
    out.writeUTF(model.getClass().getName());
    model.write(out);
  }

  /** Reads a typed Model instance from the input stream */
  public static Cluster readModel(DataInput in) throws IOException {
    String modelClassName = in.readUTF();
    Cluster model;
    try {
      model = Class.forName(modelClassName).asSubclass(Cluster.class).newInstance();
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    model.readFields(in);
    return model;
  }

  @Override
  public String asFormatString(String[] bindings) {
    return "C-" + model.getId() + ": " + model.asFormatString(bindings);
  }

  @Override
  public String asJsonString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Cluster.class, new JsonClusterModelAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, CLUSTER_TYPE);
  }

  @Override
  public int getId() {
    return model.getId();
  }

  @Override
  public Vector getCenter() {
    return model.getCenter();
  }

  @Override
  public int getNumPoints() {
    return model.getNumPoints();
  }

  @Override
  public Vector getRadius() {
    return model.getRadius();
  }

  @Override
  public void computeParameters() {
    model.computeParameters();
  }

  @Override
  public int count() {
    return model.count();
  }

  @Override
  public void observe(VectorWritable x) {
    model.observe(x);
  }

  @Override
  public double pdf(VectorWritable x) {
    return model.pdf(x);
  }

  @Override
  public Model<VectorWritable> sampleFromPosterior() {
    return model.sampleFromPosterior();
  }

}
