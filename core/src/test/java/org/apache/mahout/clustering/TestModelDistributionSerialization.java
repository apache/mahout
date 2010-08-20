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

package org.apache.mahout.clustering;

import org.apache.mahout.clustering.dirichlet.models.AbstractVectorModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.DistanceMeasureClusterDistribution;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class TestModelDistributionSerialization extends MahoutTestCase {

  public void testGaussianClusterDistribution() {
    GaussianClusterDistribution dist = new GaussianClusterDistribution(new VectorWritable(new DenseVector(2)));
    String json = dist.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(ModelDistribution.class, new JsonModelDistributionAdapter());
    builder.registerTypeAdapter(DistanceMeasure.class, new JsonDistanceMeasureAdapter());
    Gson gson = builder.create();
    GaussianClusterDistribution dist1 = (GaussianClusterDistribution) gson
        .fromJson(json, AbstractVectorModelDistribution.MODEL_DISTRIBUTION_TYPE);
    assertEquals("prototype", dist.getModelPrototype().getClass(), dist1.getModelPrototype().getClass());
  }

  public void testDMClusterDistribution() {
    DistanceMeasureClusterDistribution dist = new DistanceMeasureClusterDistribution(new VectorWritable(new DenseVector(2)));
    String json = dist.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(ModelDistribution.class, new JsonModelDistributionAdapter());
    builder.registerTypeAdapter(DistanceMeasure.class, new JsonDistanceMeasureAdapter());
    Gson gson = builder.create();
    DistanceMeasureClusterDistribution dist1 = (DistanceMeasureClusterDistribution) gson
        .fromJson(json, AbstractVectorModelDistribution.MODEL_DISTRIBUTION_TYPE);
    assertEquals("prototype", dist.getModelPrototype().getClass(), dist1.getModelPrototype().getClass());
    assertEquals("measure", dist.getMeasure().getClass(), dist1.getMeasure().getClass());
  }

  public void testDMClusterDistribution2() {
    DistanceMeasureClusterDistribution dist = new DistanceMeasureClusterDistribution(new VectorWritable(new DenseVector(2)),
                                                                                     new EuclideanDistanceMeasure());
    String json = dist.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(ModelDistribution.class, new JsonModelDistributionAdapter());
    builder.registerTypeAdapter(DistanceMeasure.class, new JsonDistanceMeasureAdapter());
    Gson gson = builder.create();
    DistanceMeasureClusterDistribution dist1 = (DistanceMeasureClusterDistribution) gson
        .fromJson(json, AbstractVectorModelDistribution.MODEL_DISTRIBUTION_TYPE);
    assertEquals("prototype", dist.getModelPrototype().getClass(), dist1.getModelPrototype().getClass());
    assertEquals("measure", dist.getMeasure().getClass(), dist1.getMeasure().getClass());
  }
}
