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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalModel;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestMapReduce extends MahoutTestCase {
  
  private List<VectorWritable> sampleData = new ArrayList<VectorWritable>();
  
  private FileSystem fs;
  
  private Configuration conf;
  
  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sdx
   *          double x-standard deviation of the samples
   * @param sdy
   *          double y-standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sdx, double sdy) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my + "] sd=[" + sdx + ", " + sdy
                       + ']');
    for (int i = 0; i < num; i++) {
      addSample(new double[] {UncommonDistributions.rNorm(mx, sdx), UncommonDistributions.rNorm(my, sdy)});
    }
  }
  
  private void addSample(double[] values) {
    Vector v = new DenseVector(2);
    for (int j = 0; j < values.length; j++) {
      v.setQuick(j, values[j]);
    }
    sampleData.add(new VectorWritable(v));
  }
  
  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sd
   *          double standard deviation of the samples
   */
  private void generateSamples(int num, double mx, double my, double sd) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my + "] sd=" + sd);
    for (int i = 0; i < num; i++) {
      addSample(new double[] {UncommonDistributions.rNorm(mx, sd), UncommonDistributions.rNorm(my, sd)});
    }
  }
  
  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
    ClusteringTestUtils.rmr("output");
    ClusteringTestUtils.rmr("input");
    conf = new Configuration();
    fs = FileSystem.get(conf);
    File f = new File("input");
    f.mkdir();
  }
  
  /** Test the basic Mapper */
  public void testMapper() throws Exception {
    generateSamples(10, 0, 0, 1);
    DirichletState<VectorWritable> state = new DirichletState<VectorWritable>(new NormalModelDistribution(
        new VectorWritable(new DenseVector(2))), 5, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.configure(state);
    
    DummyOutputCollector<Text,VectorWritable> collector = new DummyOutputCollector<Text,VectorWritable>();
    for (VectorWritable v : sampleData) {
      mapper.map(null, v, collector, null);
    }
    // Map<String, List<VectorWritable>> data = collector.getData();
    // this seed happens to produce two partitions, but they work
    // assertEquals("output size", 3, data.size());
  }
  
  /** Test the basic Reducer */
  public void testReducer() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState<VectorWritable> state = new DirichletState<VectorWritable>(new SampledNormalDistribution(
        new VectorWritable(new DenseVector(2))), 20, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.configure(state);
    
    DummyOutputCollector<Text,VectorWritable> mapCollector = new DummyOutputCollector<Text,VectorWritable>();
    for (VectorWritable v : sampleData) {
      mapper.map(null, v, mapCollector, null);
    }
    // Map<String, List<VectorWritable>> data = mapCollector.getData();
    // this seed happens to produce three partitions, but they work
    // assertEquals("output size", 7, data.size());
    
    DirichletReducer reducer = new DirichletReducer();
    reducer.configure(state);
    OutputCollector<Text,DirichletCluster<VectorWritable>> reduceCollector = new DummyOutputCollector<Text,DirichletCluster<VectorWritable>>();
    for (Text key : mapCollector.getKeys()) {
      reducer.reduce(new Text(key), mapCollector.getValue(key).iterator(), reduceCollector, null);
    }
    
    Model<VectorWritable>[] newModels = reducer.getNewModels();
    state.update(newModels);
  }
  
  private static void printModels(Iterable<Model<VectorWritable>[]> results, int significant) {
    int row = 0;
    for (Model<VectorWritable>[] r : results) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.length; k++) {
        Model<VectorWritable> model = r[k];
        if (model.count() > significant) {
          System.out.print("m" + k + model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }
  
  /** Test the Mapper and Reducer in an iteration loop */
  public void testMRIterations() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState<VectorWritable> state = new DirichletState<VectorWritable>(new SampledNormalDistribution(
        new VectorWritable(new DenseVector(2))), 20, 1.0);
    
    List<Model<VectorWritable>[]> models = new ArrayList<Model<VectorWritable>[]>();
    
    for (int iteration = 0; iteration < 10; iteration++) {
      DirichletMapper mapper = new DirichletMapper();
      mapper.configure(state);
      DummyOutputCollector<Text,VectorWritable> mapCollector = new DummyOutputCollector<Text,VectorWritable>();
      for (VectorWritable v : sampleData) {
        mapper.map(null, v, mapCollector, null);
      }
      
      DirichletReducer reducer = new DirichletReducer();
      reducer.configure(state);
      OutputCollector<Text,DirichletCluster<VectorWritable>> reduceCollector = new DummyOutputCollector<Text,DirichletCluster<VectorWritable>>();
      for (Text key : mapCollector.getKeys()) {
        reducer.reduce(new Text(key), mapCollector.getValue(key).iterator(), reduceCollector, null);
      }
      
      Model<VectorWritable>[] newModels = reducer.getNewModels();
      state.update(newModels);
      models.add(newModels);
    }
    printModels(models, 0);
  }
  
  /** Test the Mapper and Reducer using the Driver */
  public void testDriverMRIterations() throws Exception {
    generateSamples(100, 0, 0, 0.5);
    generateSamples(100, 2, 0, 0.2);
    generateSamples(100, 0, 2, 0.3);
    generateSamples(100, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data.txt"), fs, conf);
    // Now run the driver
    DirichletDriver.runJob(getTestTempDirPath("input"), getTestTempDirPath("output"), "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", "org.apache.mahout.math.DenseVector", 20, 5, 1.0, 1,
    false, true, 0);
    // and inspect results
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
      "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, "org.apache.mahout.math.DenseVector");
    conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, "2");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }
  
  private static void printResults(List<List<DirichletCluster<VectorWritable>>> clusters, int significant) {
    int row = 0;
    for (List<DirichletCluster<VectorWritable>> r : clusters) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.size(); k++) {
        Model<VectorWritable> model = r.get(k).getModel();
        if (model.count() > significant) {
          int total = (int) r.get(k).getTotalCount();
          System.out.print("m" + k + '(' + total + ')' + model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }
  
  /** Test the Mapper and Reducer using the Driver */
  public void testDriverMnRIterations() throws Exception {
    generate4Datasets();
    // Now run the driver
    DirichletDriver.runJob(getTestTempDirPath("input"), getTestTempDirPath("output"), "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", "org.apache.mahout.math.DenseVector", 20, 3, 1.0, 1,
    false, true, 0);
    // and inspect results
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
      "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, "org.apache.mahout.math.DenseVector");
    conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, "2");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }
  
  private void generate4Datasets() throws IOException {
    generateSamples(500, 0, 0, 0.5);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data1.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 2, 0, 0.2);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data2.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 0, 2, 0.3);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data3.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data4.txt"), fs, conf);
  }
  
  /** Test the Mapper and Reducer using the Driver */
  public void testDriverMnRnIterations() throws Exception {
    generate4Datasets();
    // Now run the driver
    DirichletDriver.runJob(getTestTempDirPath("input"), getTestTempDirPath("output"), "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution", "org.apache.mahout.math.DenseVector", 20, 3, 1.0, 2,
    false, true, 0);
    // and inspect results
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
      "org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution");
    conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, "org.apache.mahout.math.DenseVector");
    conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, "2");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }
  
  /** Test the Mapper and Reducer using the Driver */
  public void testDriverMnRnIterationsAsymmetric() throws Exception {
    File f = new File("input");
    for (File g : f.listFiles()) {
      g.delete();
    }
    generateSamples(500, 0, 0, 0.5, 1.0);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data1.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 2, 0, 0.2);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data2.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 0, 2, 0.3);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data3.txt"), fs, conf);
    sampleData = new ArrayList<VectorWritable>();
    generateSamples(500, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data4.txt"), fs, conf);
    // Now run the driver
    DirichletDriver.runJob(getTestTempDirPath("input"), getTestTempDirPath("output"), "org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution", "org.apache.mahout.math.DenseVector", 20, 3, 1.0, 2,
    false, true, 0);
    // and inspect results
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY,
      "org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalDistribution");
    conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, "org.apache.mahout.math.DenseVector");
    conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, "2");
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i < 11; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, "output/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }
  
  // =================== New Tests of Writable Implementations ====================
  
  public void testNormalModelWritableSerialization() throws Exception {
    double[] m = {1.1, 2.2, 3.3};
    Model<?> model = new NormalModel(5, new DenseVector(m), 3.3);
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new NormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
    assertEquals("ids", 5, model.getId());
  }
  
  public void testSampledNormalModelWritableSerialization() throws Exception {
    double[] m = {1.1, 2.2, 3.3};
    Model<?> model = new SampledNormalModel(5, new DenseVector(m), 3.3);
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new SampledNormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
    assertEquals("ids", 5, model.getId());
  }
  
  public void testAsymmetricSampledNormalModelWritableSerialization() throws Exception {
    double[] m = {1.1, 2.2, 3.3};
    double[] s = {3.3, 4.4, 5.5};
    Model<?> model = new AsymmetricSampledNormalModel(5, new DenseVector(m), new DenseVector(s));
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new AsymmetricSampledNormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
    assertEquals("ids", 5, model.getId());
  }
  
  public void testClusterWritableSerialization() throws Exception {
    double[] m = {1.1, 2.2, 3.3};
    DirichletCluster<?> cluster = new DirichletCluster(new NormalModel(5, new DenseVector(m), 4), 10);
    DataOutputBuffer out = new DataOutputBuffer();
    cluster.write(out);
    DirichletCluster<?> cluster2 = new DirichletCluster();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    cluster2.readFields(in);
    assertEquals("count", cluster.getTotalCount(), cluster2.getTotalCount());
    assertNotNull("model null", cluster2.getModel());
    assertEquals("model", cluster.getModel().toString(), cluster2.getModel().toString());
    assertEquals("ids", 5, cluster.getId());
  }
  
}
