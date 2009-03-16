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

import java.lang.reflect.Type;

import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.matrix.Vector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class DirichletCluster<Observation> {

  public Model<Observation> model; // the model for this iteration

  public double totalCount; // total count of observations for the model

  public DirichletCluster(Model<Observation> model, double totalCount) {
    super();
    this.model = model;
    this.totalCount = totalCount;
  }

  public DirichletCluster() {
    super();
  }

  public void setModel(Model<Observation> model) {
    this.model = model;
    this.totalCount += model.count();
  }

  static Type typeOfModel = new TypeToken<DirichletCluster<Vector>>() {
  }.getType();

  public String asFormatString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, typeOfModel);
  }

  @SuppressWarnings("unchecked")
  public static DirichletCluster fromFormatString(String formatString) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    return gson.fromJson(formatString, typeOfModel);
  }

}
