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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.matrix.JsonVectorAdapter;
import org.apache.mahout.matrix.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;

public class DirichletCluster<O> implements Writable {

  @SuppressWarnings("unchecked")
  @Override
  public void readFields(DataInput in) throws IOException {
    this.totalCount = in.readDouble();
    this.model = (Model<O>) readModel(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(totalCount);
    writeModel(out, model);
  }

  public Model<O> model; // the model for this iteration

  public double totalCount; // total count of observations for the model

  public DirichletCluster(Model<O> model, double totalCount) {
    super();
    this.model = model;
    this.totalCount = totalCount;
  }

  public DirichletCluster() {
    super();
  }

  public void setModel(Model<O> model) {
    this.model = model;
    this.totalCount += model.count();
  }

  private static final Type typeOfModel = new TypeToken<DirichletCluster<Vector>>() {
  }.getType();

  public String asFormatString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, typeOfModel);
  }

  /** Reads a typed Model instance from the input stream */
  @SuppressWarnings("unchecked")
  public static Model<?> readModel(DataInput in) throws IOException {
    String modelClassName = in.readUTF();
    Model<?> model;
    try {
      model = Class.forName(modelClassName).asSubclass(Model.class)
          .newInstance();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
    model.readFields(in);
    return model;
  }

  /** Writes a typed Model instance to the output stream */
  @SuppressWarnings("unchecked")
  public static void writeModel(DataOutput out, Model<?> model) throws IOException {
    out.writeUTF(model.getClass().getName());
    model.write(out);
  }

  public static DirichletCluster<Vector> fromFormatString(String formatString) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.fromJson(formatString, typeOfModel);
  }

}
