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
import org.apache.mahout.matrix.Vector;

import java.lang.reflect.Type;

@SuppressWarnings("unchecked")
public class JsonModelHolderAdapter implements JsonSerializer<ModelHolder<?>>,
    JsonDeserializer<ModelHolder<?>> {

  final Type typeOfModel = new TypeToken<Model<Vector>>() {
  }.getType();

  @Override
  public JsonElement serialize(ModelHolder<?> src, Type typeOfSrc,
                               JsonSerializationContext context) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    JsonObject obj = new JsonObject();
    obj.add("model", new JsonPrimitive(gson.toJson(src.model, typeOfModel)));
    return obj;
  }

  @Override
  public ModelHolder<?> deserialize(JsonElement json, Type typeOfT,
                                    JsonDeserializationContext context) throws JsonParseException {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    JsonObject obj = json.getAsJsonObject();
    String value = obj.get("model").getAsString();
    Model<?> m = (Model<?>) gson.fromJson(value, typeOfModel);
    return new ModelHolder(m);
  }

}
