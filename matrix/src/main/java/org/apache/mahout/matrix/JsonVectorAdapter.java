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
package org.apache.mahout.matrix;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;

public class JsonVectorAdapter implements JsonSerializer<Vector>,
    JsonDeserializer<Vector> {

  private static final Logger log = LoggerFactory.getLogger(JsonVectorAdapter.class);
  public static final String VECTOR = "vector";

  @Override
  public JsonElement serialize(Vector src, Type typeOfSrc,
                               JsonSerializationContext context) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    JsonObject obj = new JsonObject();
    obj.add(JsonMatrixAdapter.CLASS, new JsonPrimitive(src.getClass().getName()));
    obj.add(VECTOR, new JsonPrimitive(gson.toJson(src)));
    return obj;
  }

  @Override
  public Vector deserialize(JsonElement json, Type typeOfT,
                            JsonDeserializationContext context) throws JsonParseException {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    JsonObject obj = json.getAsJsonObject();
    String klass = obj.get(JsonMatrixAdapter.CLASS).getAsString();
    String vector = obj.get(VECTOR).getAsString();
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = null;
    try {
      cl = ccl.loadClass(klass);
    } catch (ClassNotFoundException e) {
      log.warn("Error while loading class", e);
    }
    return (Vector) gson.fromJson(vector, cl);
  }

}
