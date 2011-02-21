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

import java.lang.reflect.Type;

import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.dirichlet.JsonClusterModelAdapter;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.L1Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalModel;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.junit.Test;

public final class TestClusterInterface extends MahoutTestCase {

  private static final Type MODEL_TYPE = new TypeToken<Model<Vector>>() {}.getType();
  private static final Type CLUSTER_TYPE = new TypeToken<DirichletCluster>() {}.getType();
  private static final DistanceMeasure measure = new ManhattanDistanceMeasure();

  @Test
  public void testDirichletNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster model = new NormalModel(5, m, 0.75);
    String format = model.asFormatString(null);
    assertEquals("nm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    NormalModel model2 = gson.fromJson(json, MODEL_TYPE);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  @Test
  public void testDirichletSampledNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster model = new SampledNormalModel(5, m, 0.75);
    String format = model.asFormatString(null);
    assertEquals("snm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    SampledNormalModel model2 = gson.fromJson(json, MODEL_TYPE);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  @Test
  public void testDirichletASNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster model = new AsymmetricSampledNormalModel(5, m, m);
    String format = model.asFormatString(null);
    assertEquals("asnm{n=0 m=[1.100, 2.200, 3.300] sd=[1.100, 2.200, 3.300]}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    AsymmetricSampledNormalModel model2 = gson.fromJson(json, MODEL_TYPE);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  @Test
  public void testDirichletL1Model() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster model = new L1Model(5, m);
    String format = model.asFormatString(null);
    assertEquals("l1m{n=0 c=[1.100, 2.200, 3.300]}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    L1Model model2 = gson.fromJson(json, MODEL_TYPE);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  @Test
  public void testDirichletNormalModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    NormalModel model = new NormalModel(5, m, 0.75);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("C-5: nm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
  }

  @Test
  public void testDirichletNormalModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    NormalModel model = new NormalModel(5, m, 0.75);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String json = cluster.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Cluster.class, new JsonClusterModelAdapter());
    Gson gson = builder.create();
    DirichletCluster result = gson.fromJson(json, CLUSTER_TYPE);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  @Test
  public void testDirichletAsymmetricSampledNormalModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    AsymmetricSampledNormalModel model = new AsymmetricSampledNormalModel(5, m, m);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("C-5: asnm{n=0 m=[1.100, 2.200, 3.300] sd=[1.100, 2.200, 3.300]}", format);
  }

  @Test
  public void testDirichletAsymmetricSampledNormalModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    AsymmetricSampledNormalModel model = new AsymmetricSampledNormalModel(5, m, m);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String json = cluster.asJsonString();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Cluster.class, new JsonClusterModelAdapter());
    Gson gson = builder.create();
    DirichletCluster result = gson.fromJson(json, CLUSTER_TYPE);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  @Test
  public void testDirichletL1ModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    L1Model model = new L1Model(5, m);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("C-5: l1m{n=0 c=[1.100, 2.200, 3.300]}", format);
  }

  @Test
  public void testDirichletL1ModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    L1Model model = new L1Model(5, m);
    Cluster cluster = new DirichletCluster(model, 35.0);
    String json = cluster.asJsonString();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Cluster.class, new JsonClusterModelAdapter());
    Gson gson = builder.create();
    DirichletCluster result = gson.fromJson(json, CLUSTER_TYPE);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  @Test
  public void testCanopyAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[1.100, 2.200, 3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String[] bindings = { "fee", null, null };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("C-123{n=0 c=[fee:1.100, 1:2.200, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Cluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[1.100, 2.200, 3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Cluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Cluster(m, 123, measure);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("CL-123{n=0 c=[fee:1.100, 1:2.200, foo:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Cluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testMSCanopyAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new MeanShiftCanopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("MSC-123{n=0 c=[1.100, 2.200, 3.300] r=[]}", formatString);
  }

  @Test
  public void testMSCanopyAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new MeanShiftCanopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("MSC-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testMSCanopyAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new MeanShiftCanopy(m, 123, measure);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("MSC-123{n=0 c=[fee:1.100, 1:2.200, foo:3.300] r=[]}", formatString);
  }

  @Test
  public void testMSCanopyAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new MeanShiftCanopy(m, 123, measure);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("MSC-123{n=0 c=[fee:1.100, foo:3.300] r=[]}", formatString);
  }

}
