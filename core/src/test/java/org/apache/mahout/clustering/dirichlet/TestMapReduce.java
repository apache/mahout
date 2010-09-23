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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.models.AbstractVectorModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.AsymmetricSampledNormalModel;
import org.apache.mahout.clustering.dirichlet.models.NormalModel;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalDistribution;
import org.apache.mahout.clustering.dirichlet.models.SampledNormalModel;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestMapReduce extends MahoutTestCase {

  private Collection<VectorWritable> sampleData = new ArrayList<VectorWritable>();

  private FileSystem fs;

  private Configuration conf;

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
      addSample(new double[] { UncommonDistributions.rNorm(mx, sd), UncommonDistributions.rNorm(my, sd) });
    }
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = new Configuration();
    fs = FileSystem.get(conf);
  }

  /** Test the basic Mapper */
  @Test
  public void testMapper() throws Exception {
    generateSamples(10, 0, 0, 1);
    DirichletState state = new DirichletState(new NormalModelDistribution(new VectorWritable(new DenseVector(2))), 5, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.setup(state);

    RecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context context = DummyRecordWriter.build(mapper,
                                                                                                                  conf,
                                                                                                                  writer);
    for (VectorWritable v : sampleData) {
      mapper.map(null, v, context);
    }
    // Map<String, List<VectorWritable>> data = collector.getData();
    // this seed happens to produce two partitions, but they work
    // assertEquals("output size", 3, data.size());
  }

  /** Test the basic Reducer */
  @Test
  public void testReducer() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState state = new DirichletState(new SampledNormalDistribution(new VectorWritable(new DenseVector(2))), 20, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.setup(state);

    DummyRecordWriter<Text, VectorWritable> mapWriter = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                     conf,
                                                                                                                     mapWriter);
    for (VectorWritable v : sampleData) {
      mapper.map(null, v, mapContext);
    }

    DirichletReducer reducer = new DirichletReducer();
    reducer.setup(state);
    RecordWriter<Text, DirichletCluster> reduceWriter = new DummyRecordWriter<Text, DirichletCluster>();
    Reducer<Text, VectorWritable, Text, DirichletCluster>.Context reduceContext = DummyRecordWriter.build(reducer,
                                                                                                          conf,
                                                                                                          reduceWriter,
                                                                                                          Text.class,
                                                                                                          VectorWritable.class);
    for (Text key : mapWriter.getKeys()) {
      reducer.reduce(new Text(key), mapWriter.getValue(key), reduceContext);
    }

    Cluster[] newModels = reducer.getNewModels();
    state.update(newModels);
  }

  /** Test the Mapper and Reducer in an iteration loop */
  @Test
  public void testMRIterations() throws Exception {
    generateSamples(100, 0, 0, 1);
    generateSamples(100, 2, 0, 1);
    generateSamples(100, 0, 2, 1);
    generateSamples(100, 2, 2, 1);
    DirichletState state = new DirichletState(new SampledNormalDistribution(new VectorWritable(new DenseVector(2))), 20, 1.0);

    Collection<Model<VectorWritable>[]> models = new ArrayList<Model<VectorWritable>[]>();

    for (int iteration = 0; iteration < 10; iteration++) {
      DirichletMapper mapper = new DirichletMapper();
      mapper.setup(state);
      DummyRecordWriter<Text, VectorWritable> mapWriter = new DummyRecordWriter<Text, VectorWritable>();
      Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                       conf,
                                                                                                                       mapWriter);
      for (VectorWritable v : sampleData) {
        mapper.map(null, v, mapContext);
      }

      DirichletReducer reducer = new DirichletReducer();
      reducer.setup(state);
      RecordWriter<Text, DirichletCluster> reduceWriter = new DummyRecordWriter<Text, DirichletCluster>();
      Reducer<Text, VectorWritable, Text, DirichletCluster>.Context reduceContext = DummyRecordWriter.build(reducer,
                                                                                                            conf,
                                                                                                            reduceWriter,
                                                                                                            Text.class,
                                                                                                            VectorWritable.class);
      for (Text key : mapWriter.getKeys()) {
        reducer.reduce(new Text(key), mapWriter.getValue(key), reduceContext);
      }

      Cluster[] newModels = reducer.getNewModels();
      state.update(newModels);
      models.add(newModels);
    }
    printModels(models, 0);
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

  private static void printResults(Iterable<List<DirichletCluster>> clusters, int significant) {
    int row = 0;
    for (List<DirichletCluster> r : clusters) {
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

  /** Test the Mapper and Reducer using the Driver in sequential execution mode */
  @Test
  public void testDriverIterationsSeq() throws Exception {
    generateSamples(100, 0, 0, 0.5);
    generateSamples(100, 2, 0, 0.2);
    generateSamples(100, 0, 2, 0.3);
    generateSamples(100, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data.txt"), fs, conf);
    // Now run the driver using the run() method. Others can use runJob() as before
    Integer maxIterations = 5;
    AbstractVectorModelDistribution modelDistribution = new SampledNormalDistribution(new VectorWritable(new DenseVector(2)));
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), modelDistribution.getClass().getName(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), modelDistribution.getModelPrototype().get().getClass().getName(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0", optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION), optKey(DefaultOptionCreator.METHOD_OPTION),
        DefaultOptionCreator.SEQUENTIAL_METHOD };
    new DirichletDriver().run(args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = new ArrayList<List<DirichletCluster>>();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, modelDistribution.asJsonString());
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i <= maxIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, new Path(getTestTempDirPath("output"), "clusters-" + i).toString());
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }

  /** Test the Mapper and Reducer using the Driver in mapreduce mode */
  @Test
  public void testDriverIterationsMR() throws Exception {
    generateSamples(100, 0, 0, 0.5);
    generateSamples(100, 2, 0, 0.2);
    generateSamples(100, 0, 2, 0.3);
    generateSamples(100, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data.txt"), fs, conf);
    // Now run the driver using the run() method. Others can use runJob() as before
    Integer maxIterations = 5;
    AbstractVectorModelDistribution modelDistribution = new SampledNormalDistribution(new VectorWritable(new DenseVector(2)));
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), modelDistribution.getClass().getName(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), modelDistribution.getModelPrototype().get().getClass().getName(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0", optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION) };
    ToolRunner.run(new Configuration(), new DirichletDriver(), args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = new ArrayList<List<DirichletCluster>>();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, modelDistribution.asJsonString());
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i <= maxIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, new Path(getTestTempDirPath("output"), "clusters-" + i).toString());
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }

  /** Test the Mapper and Reducer using the Driver */
  @Test
  public void testDriverMnRIterations() throws Exception {
    generate4Datasets();
    // Now run the driver
    int maxIterations = 3;
    AbstractVectorModelDistribution modelDistribution = new SampledNormalDistribution(new VectorWritable(new DenseVector(2)));
    Configuration conf = new Configuration();
    DirichletDriver.run(conf,
                        getTestTempDirPath("input"),
                        getTestTempDirPath("output"),
                        modelDistribution,
                        20,
                        maxIterations,
                        1.0,
                        false,
                        true,
                        0,
                        false);
    // and inspect results
    List<List<DirichletCluster>> clusters = new ArrayList<List<DirichletCluster>>();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, modelDistribution.asJsonString());
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i <= maxIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, new Path(getTestTempDirPath("output"), "clusters-" + i).toString());
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

  @Test
  public void testNormalModelWritableSerialization() throws Exception {
    double[] m = { 1.1, 2.2, 3.3 };
    Model<?> model = new NormalModel(5, new DenseVector(m), 3.3);
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new NormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
  }

  @Test
  public void testSampledNormalModelWritableSerialization() throws Exception {
    double[] m = { 1.1, 2.2, 3.3 };
    Model<?> model = new SampledNormalModel(5, new DenseVector(m), 3.3);
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new SampledNormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
  }

  @Test
  public void testAsymmetricSampledNormalModelWritableSerialization() throws Exception {
    double[] m = { 1.1, 2.2, 3.3 };
    double[] s = { 3.3, 4.4, 5.5 };
    Model<?> model = new AsymmetricSampledNormalModel(5, new DenseVector(m), new DenseVector(s));
    DataOutputBuffer out = new DataOutputBuffer();
    model.write(out);
    Model<?> model2 = new AsymmetricSampledNormalModel();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    model2.readFields(in);
    assertEquals("models", model.toString(), model2.toString());
  }

  @Test
  public void testClusterWritableSerialization() throws Exception {
    double[] m = { 1.1, 2.2, 3.3 };
    DirichletCluster cluster = new DirichletCluster(new NormalModel(5, new DenseVector(m), 4), 10);
    DataOutputBuffer out = new DataOutputBuffer();
    cluster.write(out);
    DirichletCluster cluster2 = new DirichletCluster();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    cluster2.readFields(in);
    assertEquals("count", cluster.getTotalCount(), cluster2.getTotalCount(), EPSILON);
    assertNotNull("model null", cluster2.getModel());
    assertEquals("model", cluster.getModel().toString(), cluster2.getModel().toString());
  }

}
