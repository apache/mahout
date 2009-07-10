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

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;

public class JsonModelDistributionAdapter implements
    JsonSerializer<ModelDistribution<?>>, JsonDeserializer<ModelDistribution<?>> {

  private static final Logger log = LoggerFactory.getLogger(JsonModelDistributionAdapter.class);

  @Override
  public JsonElement serialize(ModelDistribution<?> src, Type typeOfSrc,
                               JsonSerializationContext context) {
    return new JsonPrimitive(src.getClass().getName());
  }

  @Override
  public ModelDistribution<?> deserialize(JsonElement json, Type typeOfT,
                                          JsonDeserializationContext context) throws JsonParseException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl;
    try {
      cl = ccl.loadClass(json.getAsString());
    } catch (ClassNotFoundException e) {
      log.warn("Error while loading class", e);
      return null;
    }
    try {
      return (ModelDistribution<?>) cl.newInstance();
    } catch (InstantiationException e) {
      log.warn("Error while creating class", e);
    } catch (IllegalAccessException e) {
      log.warn("Error while creating class", e);
    }
    return null;
  }

}
