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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalModel;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DummyOutputCollector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class TestMapReduce extends TestCase {

  private List<Vector> sampleData = new ArrayList<Vector>();

  /**
   * Generate random samples and add them to the sampleData
   * @param num int number of samples to generate
   * @param mx double x-value of the sample mean
   * @param my double y-value of the sample mean
   * @param sd double standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sd) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my
        + "] sd=" + sd);
    for (int i = 0; i < num; i++)
      sampleData.add(new DenseVector(new double[] {
          UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd) }));
  }

  public static void writePointsToFileWithPayload(List<Vector> points,
      String fileName, String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
        new FileOutputStream(fileName), Charset.forName("UTF-8")));
    for (Vector point : points) {
      output.write(point.asFormatString());
      output.write(payload);
      output.write('\n');
    }
    output.flush();
    output.close();
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
    File f = new File("input");
    if (!f.exists())
      f.mkdir();
  }

  /**
   * Test the basic Mapper 
   * @throws Exception
   */
  public void testMapper() throws Exception {
    generateSamples(10, 0, 0, 1);
    DirichletState<Vector> state = new DirichletState<Vector>(
        new NormalModelDistribution(), 5, 1, 0, 0);
    DirichletMapper mapper = new DirichletMapper();
    mapper.configure(state);

    DummyOutputCollector<Text, Text> collector = new DummyOutputCollector<Text, Text>();
    for (Vector v : sampleData)
      mapper.map(null, new Text(v.asFormatString()), collector, null);
    Map<String, List<Text>> data = collector.getData();
    // this seed happens to produce two partitions, but they work
    assertEquals("output size", 3, data.size());
  }

  /**
   * Test the basic Reducer 
   * @throws Exception
   */
  public void testReducer() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState<Vector> state = new DirichletState<Vector>(
        new SampledNormalDistribution(), 20, 1, 1, 0);
    DirichletMapper mapper = new DirichletMapper();
    mapper.configure(state);

    DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
    for (Vector v : sampleData)
      mapper.map(null, new Text(v.asFormatString()), mapCollector, null);
    Map<String, List<Text>> data = mapCollector.getData();
    // this seed happens to produce three partitions, but they work
    assertEquals("output size", 7, data.size());

    DirichletReducer reducer = new DirichletReducer();
    reducer.configure(state);
    DummyOutputCollector<Text, Text> reduceCollector = new DummyOutputCollector<Text, Text>();
    for (String key : mapCollector.getKeys())
      reducer.reduce(new Text(key), mapCollector.getValue(key).iterator(),
          reduceCollector, null);

    Model<Vector>[] newModels = reducer.newModels;
    state.update(newModels);
  }

  private void printModels(List<Model<Vector>[]> results, int significant) {
    int row = 0;
    for (Model<Vector>[] r : results) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.length; k++) {
        Model<Vector> model = r[k];
        if (model.count() > significant) {
          System.out.print("m" + k + model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  /**
   * Test the Mapper and Reducer in an iteration loop 
   * @throws Exception
   */
  public void testMRIterations() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState<Vector> state = new DirichletState<Vector>(
        new SampledNormalDistribution(), 20, 1.0, 1, 0);

    List<Model<Vector>[]> models = new ArrayList<Model<Vector>[]>();

    for (int iteration = 0; iteration < 10; iteration++) {
      DirichletMapper mapper = new DirichletMapper();
      mapper.configure(state);
      DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
      for (Vector v : sampleData)
        mapper.map(null, new Text(v.asFormatString()), mapCollector, null);

      DirichletReducer reducer = new DirichletReducer();
      reducer.configure(state);
      DummyOutputCollector<Text, Text> reduceCollector = new DummyOutputCollector<Text, Text>();
      for (String key : mapCollector.getKeys())
        reducer.reduce(new Text(key), mapCollector.getValue(key).iterator(),
            reduceCollector, null);

      Model<Vector>[] newModels = reducer.newModels;
      state.update(newModels);
      models.add(newModels);
    }
    printModels(models, 0);
  }

  @SuppressWarnings("unchecked")
  public void testNormalModelSerialization() {
    double[] m = { 1.1, 2.2 };
    Model model = new NormalModel(new DenseVector(m), 3.3);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(model);
    Model model2 = gson.fromJson(jsonString, NormalModel.class);
    assertEquals("models", model.toString(), model2.toString());
  }

  @SuppressWarnings("unchecked")
  public void testNormalModelDistributionSerialization() {
    NormalModelDistribution dist = new NormalModelDistribution();
    Model[] models = dist.sampleFromPrior(20);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(models);
    Model[] models2 = gson.fromJson(jsonString, NormalModel[].class);
    assertEquals("models", models.length, models2.length);
    for (int i = 0; i < models.length; i++)
      assertEquals("model[" + i + "]", models[i].toString(), models2[i]
          .toString());
  }

  @SuppressWarnings("unchecked")
  public void testSampledNormalModelSerialization() {
    double[] m = { 1.1, 2.2 };
    Model model = new SampledNormalModel(new DenseVector(m), 3.3);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(model);
    Model model2 = gson.fromJson(jsonString, SampledNormalModel.class);
    assertEquals("models", model.toString(), model2.toString());
  }

  @SuppressWarnings("unchecked")
  public void testSampledNormalDistributionSerialization() {
    SampledNormalDistribution dist = new SampledNormalDistribution();
    Model[] models = dist.sampleFromPrior(20);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(models);
    Model[] models2 = gson.fromJson(jsonString, SampledNormalModel[].class);
    assertEquals("models", models.length, models2.length);
    for (int i = 0; i < models.length; i++)
      assertEquals("model[" + i + "]", models[i].toString(), models2[i]
          .toString());
  }

  @SuppressWarnings("unchecked")
  public void testAsymmetricSampledNormalModelSerialization() {
    double[] m = { 1.1, 2.2 };
    double[] s = { 3.3, 4.4 };
    Model model = new AsymmetricSampledNormalModel(new DenseVector(m),
        new DenseVector(s));
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(model);
    Model model2 = gson
        .fromJson(jsonString, AsymmetricSampledNormalModel.class);
    assertEquals("models", model.toString(), model2.toString());
  }

  @SuppressWarnings("unchecked")
  public void testAsymmetricSampledNormalDistributionSerialization() {
    AsymmetricSampledNormalDistribution dist = new AsymmetricSampledNormalDistribution();
    Model[] models = dist.sampleFromPrior(20);
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    String jsonString = gson.toJson(models);
    Model[] models2 = gson.fromJson(jsonString,
        AsymmetricSampledNormalModel[].class);
    assertEquals("models", models.length, models2.length);
    for (int i = 0; i < models.length; i++)
      assertEquals("model[" + i + "]", models[i].toString(), models2[i]
          .toString());
  }

  @SuppressWarnings("unchecked")
  public void testModelHolderSerialization() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder
        .registerTypeAdapter(ModelHolder.class, new JsonModelHolderAdapter());
    Gson gson = builder.create();
    double[] d = { 1.1, 2.2 };
    ModelHolder mh = new ModelHolder(new NormalModel(new DenseVector(d), 3.3));
    String format = gson.toJson(mh);
    System.out.println(format);
    ModelHolder mh2 = gson.fromJson(format, ModelHolder.class);
    assertEquals("mh", mh.model.toString(), mh2.model.toString());
  }

  @SuppressWarnings("unchecked")
  public void testModelHolderSerialization2() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder
        .registerTypeAdapter(ModelHolder.class, new JsonModelHolderAdapter());
    Gson gson = builder.create();
    double[] d = { 1.1, 2.2 };
    double[] s = { 3.3, 4.4 };
    ModelHolder mh = new ModelHolder(new AsymmetricSampledNormalModel(
        new DenseVector(d), new DenseVector(s)));
    String format = gson.toJson(mh);
    System.out.println(format);
    ModelHolder mh2 = gson.fromJson(format, ModelHolder.class);
    assertEquals("mh", mh.model.toString(), mh2.model.toString());
  }

  @SuppressWarnings("unchecked")
  public void testStateSerialization() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(DirichletState.class,
        new JsonDirichletStateAdapter());
    Gson gson = builder.create();
    DirichletState state = new DirichletState(new SampledNormalDistribution(),
        20, 1, 1, 0);
    String format = gson.toJson(state);
    System.out.println(format);
    DirichletState state2 = gson.fromJson(format, DirichletState.class);
    assertNotNull("State2 null", state2);
    assertEquals("numClusters", state.numClusters, state2.numClusters);
    assertEquals("modelFactory", state.modelFactory.getClass().getName(),
        state2.modelFactory.getClass().getName());
    assertEquals("clusters", state.clusters.size(), state2.clusters.size());
    assertEquals("mixture", state.mixture.cardinality(), state2.mixture
        .cardinality());
    assertEquals("dirichlet", state.offset, state2.offset);
  }

  /**
   * Test the Mapper and Reducer using the Driver 
   * @throws Exception
   */
  public void testDriverMRIterations() throws Exception {
    File f = new File("input");
    for (File g : f.listFiles())
      g.delete();
    generateSamples(100, 0, 0, 0.5);
    generateSamples(100, 2, 0, 0.2);
    generateSamples(100, 0, 2, 0.3);
    generateSamples(100, 2, 2, 1);
    writePointsToFileWithPayload(sampleData, "input/data.txt", "");
    // Now run the driver
    DirichletDriver.runJob("input", "output",
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", 20,
        10, 1.0, 1);
    // and inspect results
    List<List<DirichletCluster<Vector>>> clusters = new ArrayList<List<DirichletCluster<Vector>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(20));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(1.0));
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).clusters);
    }
    printResults(clusters, 0);
  }

  private void printResults(List<List<DirichletCluster<Vector>>> clusters,
      int significant) {
    int row = 0;
    for (List<DirichletCluster<Vector>> r : clusters) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.size(); k++) {
        Model<Vector> model = r.get(k).model;
        if (model.count() > significant) {
          int total = new Double(r.get(k).totalCount).intValue();
          System.out.print("m" + k + "(" + total + ")" + model.toString()
              + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  /**
   * Test the Mapper and Reducer using the Driver 
   * @throws Exception
   */
  public void testDriverMnRIterations() throws Exception {
    File f = new File("input");
    for (File g : f.listFiles())
      g.delete();
    generateSamples(500, 0, 0, 0.5);
    writePointsToFileWithPayload(sampleData, "input/data1.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 0, 0.2);
    writePointsToFileWithPayload(sampleData, "input/data2.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 0, 2, 0.3);
    writePointsToFileWithPayload(sampleData, "input/data3.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 2, 1);
    writePointsToFileWithPayload(sampleData, "input/data4.txt", "");
    // Now run the driver
    DirichletDriver.runJob("input", "output",
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", 20,
        15, 1.0, 1);
    // and inspect results
    List<List<DirichletCluster<Vector>>> clusters = new ArrayList<List<DirichletCluster<Vector>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(20));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(1.0));
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).clusters);
    }
    printResults(clusters, 0);
  }

  /**
   * Test the Mapper and Reducer using the Driver 
   * @throws Exception
   */
  public void testDriverMnRnIterations() throws Exception {
    File f = new File("input");
    for (File g : f.listFiles())
      g.delete();
    generateSamples(500, 0, 0, 0.5);
    writePointsToFileWithPayload(sampleData, "input/data1.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 0, 0.2);
    writePointsToFileWithPayload(sampleData, "input/data2.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 0, 2, 0.3);
    writePointsToFileWithPayload(sampleData, "input/data3.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 2, 1);
    writePointsToFileWithPayload(sampleData, "input/data4.txt", "");
    // Now run the driver
    DirichletDriver.runJob("input", "output",
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", 20,
        15, 1.0, 2);
    // and inspect results
    List<List<DirichletCluster<Vector>>> clusters = new ArrayList<List<DirichletCluster<Vector>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
        "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(20));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(1.0));
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).clusters);
    }
    printResults(clusters, 0);
  }

  /**
   * Generate random samples and add them to the sampleData
   * @param num int number of samples to generate
   * @param mx double x-value of the sample mean
   * @param my double y-value of the sample mean
   * @param sdx double x-standard deviation of the samples
   * @param sdy double y-standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sdx,
      double sdy) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my
        + "] sd=[" + sdx + ", " + sdy + "]");
    for (int i = 0; i < num; i++)
      sampleData.add(new DenseVector(new double[] {
          UncommonDistributions.rNorm(mx, sdx),
          UncommonDistributions.rNorm(my, sdy) }));
  }

  /**
   * Test the Mapper and Reducer using the Driver 
   * @throws Exception
   */
  public void testDriverMnRnIterationsAsymmetric() throws Exception {
    File f = new File("input");
    for (File g : f.listFiles())
      g.delete();
    generateSamples(500, 0, 0, 0.5, 1.0);
    writePointsToFileWithPayload(sampleData, "input/data1.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 0, 0.2, 0.1);
    writePointsToFileWithPayload(sampleData, "input/data2.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 0, 2, 0.3, 0.5);
    writePointsToFileWithPayload(sampleData, "input/data3.txt", "");
    sampleData = new ArrayList<Vector>();
    generateSamples(500, 2, 2, 1, 0.5);
    writePointsToFileWithPayload(sampleData, "input/data4.txt", "");
    // Now run the driver
    DirichletDriver
        .runJob(
            "input",
            "output",
            "org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution",
            20, 15, 1.0, 2);
    // and inspect results
    List<List<DirichletCluster<Vector>>> clusters = new ArrayList<List<DirichletCluster<Vector>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf
        .set(DirichletDriver.MODEL_FACTORY_KEY,
            "org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(20));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(1.0));
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).clusters);
    }
    printResults(clusters, 0);
  }

}
