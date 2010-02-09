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

import junit.framework.TestCase;

import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.dirichlet.JsonModelAdapter;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.L1Model;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalModel;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class TestPrintableInterface extends TestCase {

  private static final Type modelType = new TypeToken<Model<Vector>>() {
  }.getType();

  private static final Type clusterType = new TypeToken<DirichletCluster<Vector>>() {
  }.getType();

  public void testDirichletNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable model = new NormalModel(m, 0.75);
    String format = model.asFormatString(null);
    assertEquals("format", "nm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    NormalModel model2 = gson.fromJson(json, modelType);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  public void testDirichletSampledNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable model = new SampledNormalModel(m, 0.75);
    String format = model.asFormatString(null);
    assertEquals("format", "snm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    SampledNormalModel model2 = gson.fromJson(json, modelType);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  public void testDirichletASNormalModel() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable model = new AsymmetricSampledNormalModel(m, m);
    String format = model.asFormatString(null);
    assertEquals("format", "asnm{n=0 m=[1.100, 2.200, 3.300] sd=[1.100, 2.200, 3.300]}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    AsymmetricSampledNormalModel model2 = gson.fromJson(json, modelType);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  public void testDirichletL1Model() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable model = new L1Model(m);
    String format = model.asFormatString(null);
    assertEquals("format", "l1m{n=0 c=[1.100, 2.200, 3.300]}", format);
    String json = model.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    L1Model model2 = gson.fromJson(json, modelType);
    assertEquals("Json", format, model2.asFormatString(null));
  }

  public void testDirichletNormalModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    NormalModel model = new NormalModel(m, 0.75);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("format", "nm{n=0 m=[1.100, 2.200, 3.300] sd=0.75}", format);
  }

  public void testDirichletNormalModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    NormalModel model = new NormalModel(m, 0.75);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String json = cluster.asJsonString();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    DirichletCluster<VectorWritable> result = gson.fromJson(json, clusterType);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  public void testDirichletAsymmetricSampledNormalModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    AsymmetricSampledNormalModel model = new AsymmetricSampledNormalModel(m, m);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("format", "asnm{n=0 m=[1.100, 2.200, 3.300] sd=[1.100, 2.200, 3.300]}", format);
  }

  public void testDirichletAsymmetricSampledNormalModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    AsymmetricSampledNormalModel model = new AsymmetricSampledNormalModel(m, m);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String json = cluster.asJsonString();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    DirichletCluster<VectorWritable> result = gson.fromJson(json, clusterType);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  public void testDirichletL1ModelClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    L1Model model = new L1Model(m);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String format = cluster.asFormatString(null);
    assertEquals("format", "l1m{n=0 c=[1.100, 2.200, 3.300]}", format);
  }

  public void testDirichletL1ModelClusterAsJsonString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    L1Model model = new L1Model(m);
    Printable cluster = new DirichletCluster<VectorWritable>(model, 35.0);
    String json = cluster.asJsonString();

    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Model.class, new JsonModelAdapter());
    Gson gson = builder.create();
    DirichletCluster<VectorWritable> result = gson.fromJson(json, clusterType);
    assertNotNull("result null", result);
    assertEquals("model", cluster.asFormatString(null), result.asFormatString(null));
  }

  public void testCanopyAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new Canopy(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [1.100, 2.200, 3.300]", formatString);
  }

  public void testCanopyAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new Canopy(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [0:1.100, 2:3.300]", formatString);
  }

  public void testCanopyAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new Canopy(m, 123);
    String[] bindings = { "fee", null, null };
    String formatString = cluster.asFormatString(bindings);
    System.out.println(formatString);
    assertEquals("format", "C123: [fee:1.100, 1:2.200, 2:3.300]", formatString);
  }

  public void testCanopyAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new Canopy(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [0:1.100, 2:3.300]", formatString);
  }

  public void testClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new Cluster(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [1.100, 2.200, 3.300]", formatString);
  }

  public void testClusterAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new Cluster(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [0:1.100, 2:3.300]", formatString);
  }

  public void testClusterAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new Cluster(m, 123);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    System.out.println(formatString);
    assertEquals("format", "C123: [fee:1.100, 1:2.200, foo:3.300]", formatString);
  }

  public void testClusterAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new Cluster(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [0:1.100, 2:3.300]", formatString);
  }

  public void testMSCanopyAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new MeanShiftCanopy(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [1.100, 2.200, 3.300]", formatString);
  }

  public void testMSCanopyAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new MeanShiftCanopy(m, 123);
    String formatString = cluster.asFormatString(null);
    System.out.println(formatString);
    assertEquals("format", "C123: [0:1.100, 2:3.300]", formatString);
  }

  public void testMSCanopyAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Printable cluster = new MeanShiftCanopy(m, 123);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    System.out.println(formatString);
    assertEquals("format", "C123: [fee:1.100, 1:2.200, foo:3.300]", formatString);
  }

  public void testMSCanopyAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Printable cluster = new MeanShiftCanopy(m, 123);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    System.out.println(formatString);
    assertEquals("format", "C123: [fee:1.100, foo:3.300]", formatString);
  }

}
