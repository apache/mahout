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
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.apache.mahout.matrix.JsonVectorAdapter;
import org.apache.mahout.matrix.Vector;

import java.lang.reflect.Type;
import java.util.List;

@SuppressWarnings("unchecked")
public class JsonDirichletStateAdapter implements
    JsonSerializer<DirichletState<?>>, JsonDeserializer<DirichletState<?>> {

  private final Type typeOfModel = new TypeToken<List<DirichletCluster<Vector>>>() {
  }.getType();

  private final Type typeOfModelDistribution = new TypeToken<ModelDistribution<Vector>>() {
  }.getType();

  @Override
  public JsonElement serialize(DirichletState<?> src, Type typeOfSrc,
                               JsonSerializationContext context) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    builder.registerTypeAdapter(ModelDistribution.class,
        new JsonModelDistributionAdapter());
    Gson gson = builder.create();
    JsonObject obj = new JsonObject();
    obj.addProperty("numClusters", src.numClusters);
    obj.addProperty("offset", src.offset);
    obj.add("modelFactory", new JsonPrimitive(gson.toJson(src.modelFactory,
        typeOfModelDistribution)));
    obj.add("clusters", new JsonPrimitive(gson
        .toJson(src.clusters, typeOfModel)));
    obj.add("mixture",
        new JsonPrimitive(gson.toJson(src.mixture, Vector.class)));
    return obj;
  }

  @Override
  public DirichletState<?> deserialize(JsonElement json, Type typeOfT,
                                       JsonDeserializationContext context) throws JsonParseException {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    builder.registerTypeAdapter(ModelDistribution.class,
        new JsonModelDistributionAdapter());
    Gson gson = builder.create();
    JsonObject obj = json.getAsJsonObject();
    DirichletState<?> state = new DirichletState();
    state.numClusters = obj.get("numClusters").getAsInt();
    state.offset = obj.get("offset").getAsDouble();
    state.modelFactory = gson.fromJson(obj.get("modelFactory").getAsString(),
        typeOfModelDistribution);
    state.clusters = gson.fromJson(obj.get("clusters").getAsString(),
        typeOfModel);
    state.mixture = gson.fromJson(obj.get("mixture").getAsString(), Vector.class);
    return state;
  }

}
