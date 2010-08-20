/* Licensed to the Apache Software Foundation (ASF) under one or more
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

import java.lang.reflect.Type;

import org.apache.mahout.clustering.JsonDistanceMeasureAdapter;
import org.apache.mahout.clustering.JsonModelDistributionAdapter;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public abstract class AbstractVectorModelDistribution implements ModelDistribution<VectorWritable> {

  public static final Type MODEL_DISTRIBUTION_TYPE = new TypeToken<ModelDistribution<VectorWritable>>() {
  }.getType();

  // a prototype instance used for creating prior model distributions using like(). It
  // should be of the class and cardinality desired for the particular application.
  private VectorWritable modelPrototype;

  protected AbstractVectorModelDistribution() {
  }

  protected AbstractVectorModelDistribution(VectorWritable modelPrototype) {
    this.modelPrototype = modelPrototype;
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.ModelDistribution#asJsonString()
   */
  @Override
  public String asJsonString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(ModelDistribution.class, new JsonModelDistributionAdapter());
    builder.registerTypeAdapter(DistanceMeasure.class, new JsonDistanceMeasureAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, MODEL_DISTRIBUTION_TYPE);
  }

  /**
   * @return the modelPrototype
   */
  public VectorWritable getModelPrototype() {
    return modelPrototype;
  }

  /**
   * @param modelPrototype
   *          the modelPrototype to set
   */
  public void setModelPrototype(VectorWritable modelPrototype) {
    this.modelPrototype = modelPrototype;
  }

}
