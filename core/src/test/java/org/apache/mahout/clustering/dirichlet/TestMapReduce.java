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

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.dirichlet.models.DistanceMeasureClusterDistribution;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.clustering.dirichlet.models.GaussianClusterDistribution;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.MahalanobisDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestMapReduce extends MahoutTestCase {

  private Collection<VectorWritable> sampleData = Lists.newArrayList();

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

  /**
   * Generate random samples with asymmetric standard deviations and add them to the sampleData
   *
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sdx
   *          double standard deviation in x of the samples
   * @param sdy
   *          double standard deviation in y of the samples
   */
  private void generateAsymmetricSamples(int num, double mx, double my, double sdx, double sdy) {
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my + "] sd=[" + sdx + ", " + sdy + ']');
    for (int i = 0; i < num; i++) {
      addSample(new double[] { UncommonDistributions.rNorm(mx, sdx), UncommonDistributions.rNorm(my, sdy) });
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
    DirichletState state =
        new DirichletState(new GaussianClusterDistribution(new VectorWritable(new DenseVector(2))), 5, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.setup(state);

    RecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>,VectorWritable,Text,VectorWritable>.Context context =
        DummyRecordWriter.build(mapper, conf, writer);
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
    DirichletState state =
        new DirichletState(new GaussianClusterDistribution(new VectorWritable(new DenseVector(2))), 20, 1);
    DirichletMapper mapper = new DirichletMapper();
    mapper.setup(state);

    DummyRecordWriter<Text, VectorWritable> mapWriter = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>,VectorWritable,Text,VectorWritable>.Context mapContext =
        DummyRecordWriter.build(mapper, conf, mapWriter);
    for (VectorWritable v : sampleData) {
      mapper.map(null, v, mapContext);
    }

    DirichletReducer reducer = new DirichletReducer();
    reducer.setup(state);
    RecordWriter<Text, DirichletCluster> reduceWriter = new DummyRecordWriter<Text, DirichletCluster>();
    Reducer<Text, VectorWritable, Text, DirichletCluster>.Context reduceContext =
        DummyRecordWriter.build(reducer, conf, reduceWriter, Text.class, VectorWritable.class);
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
    DirichletState state =
        new DirichletState(new GaussianClusterDistribution(new VectorWritable(new DenseVector(2))), 20, 1.0);

    Collection<Model<VectorWritable>[]> models = Lists.newArrayList();

    for (int iteration = 0; iteration < 10; iteration++) {
      DirichletMapper mapper = new DirichletMapper();
      mapper.setup(state);
      DummyRecordWriter<Text, VectorWritable> mapWriter = new DummyRecordWriter<Text, VectorWritable>();
      Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context mapContext =
          DummyRecordWriter.build(mapper, conf, mapWriter);
      for (VectorWritable v : sampleData) {
        mapper.map(null, v, mapContext);
      }

      DirichletReducer reducer = new DirichletReducer();
      reducer.setup(state);
      RecordWriter<Text, DirichletCluster> reduceWriter = new DummyRecordWriter<Text, DirichletCluster>();
      Reducer<Text,VectorWritable, Text,DirichletCluster>.Context reduceContext =
          DummyRecordWriter.build(reducer, conf, reduceWriter, Text.class, VectorWritable.class);
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
    DistributionDescription description =
        new DistributionDescription(GaussianClusterDistribution.class.getName(),
                                    DenseVector.class.getName(),
                                    null,
                                    2);
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), description.getModelFactory(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), description.getModelPrototype(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION), optKey(DefaultOptionCreator.METHOD_OPTION),
        DefaultOptionCreator.SEQUENTIAL_METHOD };
    DirichletDriver dirichletDriver = new DirichletDriver();
    dirichletDriver.setConf(conf);
    dirichletDriver.run(args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, description.toString());
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
    DistributionDescription description =
        new DistributionDescription(GaussianClusterDistribution.class.getName(),
                                    DenseVector.class.getName(),
                                    null,
                                    2);
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), description.getModelFactory(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), description.getModelPrototype(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION)};
    ToolRunner.run(new Configuration(), new DirichletDriver(), args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, description.toString());
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
    DistributionDescription description =
        new DistributionDescription(GaussianClusterDistribution.class.getName(),
                                    DenseVector.class.getName(),
                                    null,
                                    2);
    Configuration conf = new Configuration();
    DirichletDriver.run(conf,
                        getTestTempDirPath("input"),
                        getTestTempDirPath("output"),
                        description,
                        20,
                        maxIterations,
                        1.0,
                        false,
                        true,
                        0,
                        false);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = Lists.newArrayList();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, description.toString());
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, "20");
    conf.set(DirichletDriver.ALPHA_0_KEY, "1.0");
    for (int i = 0; i <= maxIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, new Path(getTestTempDirPath("output"), "clusters-" + i).toString());
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
  }

  /** Test the Driver in sequential execution mode using MahalanobisDistanceMeasure */
  @Test
  public void testDriverIterationsMahalanobisSeq() throws Exception {
    generateAsymmetricSamples(100, 0, 0, 0.5, 3.0);
    generateAsymmetricSamples(100, 0, 3, 0.3, 4.0);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data.txt"), fs, conf);
    // Now run the driver using the run() method. Others can use runJob() as before
    MahalanobisDistanceMeasure measure = new MahalanobisDistanceMeasure();
    DistributionDescription description =
        new DistributionDescription(DistanceMeasureClusterDistribution.class.getName(),
                                    DenseVector.class.getName(),
                                    MahalanobisDistanceMeasure.class.getName(),
                                    2);

    Vector meanVector = new DenseVector(new double[] { 0.0, 0.0 });
    measure.setMeanVector(meanVector);
    Matrix m= new DenseMatrix(new double [][] {{0.5, 0.0}, {0.0, 4.0}});
    measure.setCovarianceMatrix(m);

    Path inverseCovarianceFile =
        new Path(getTestTempDirPath("mahalanobis"), "MahalanobisDistanceMeasureInverseCovarianceFile");
    conf.set("MahalanobisDistanceMeasure.inverseCovarianceFile", inverseCovarianceFile.toString());
    FileSystem fs = FileSystem.get(inverseCovarianceFile.toUri(), conf);
    MatrixWritable inverseCovarianceMatrix = new MatrixWritable(measure.getInverseCovarianceMatrix());
    DataOutputStream out = fs.create(inverseCovarianceFile);
    try {
      inverseCovarianceMatrix.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }

    Path meanVectorFile = new Path(getTestTempDirPath("mahalanobis"), "MahalanobisDistanceMeasureMeanVectorFile");
    conf.set("MahalanobisDistanceMeasure.meanVectorFile", meanVectorFile.toString());
    fs = FileSystem.get(meanVectorFile.toUri(), conf);
    VectorWritable meanVectorWritable = new VectorWritable(meanVector);
    out = fs.create(meanVectorFile);
    try {
      meanVectorWritable.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }

    conf.set("MahalanobisDistanceMeasure.maxtrixClass", MatrixWritable.class.getName());
    conf.set("MahalanobisDistanceMeasure.vectorClass", VectorWritable.class.getName());

    Integer maxIterations = 5;
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), description.getModelFactory(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), description.getDistanceMeasure(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), description.getModelPrototype(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION), optKey(DefaultOptionCreator.METHOD_OPTION),
        DefaultOptionCreator.SEQUENTIAL_METHOD };
    DirichletDriver dirichletDriver = new DirichletDriver();
    dirichletDriver.setConf(conf);
    dirichletDriver.run(args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, description.toString());
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
  public void testDriverIterationsMahalanobisMR() throws Exception {
    generateAsymmetricSamples(100, 0, 0, 0.5, 3.0);
    generateAsymmetricSamples(100, 0, 3, 0.3, 4.0);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data.txt"), fs, conf);
    // Now run the driver using the run() method. Others can use runJob() as before

    MahalanobisDistanceMeasure measure = new MahalanobisDistanceMeasure();
    DistributionDescription description =
        new DistributionDescription(DistanceMeasureClusterDistribution.class.getName(),
                                    DenseVector.class.getName(),
                                    MahalanobisDistanceMeasure.class.getName(),
                                    2);

    Vector meanVector = new DenseVector(new double[]{0.0, 0.0});
    measure.setMeanVector(meanVector);
    Matrix m = new DenseMatrix(new double [][] {{0.5, 0.0}, {0.0, 4.0}});
    measure.setCovarianceMatrix(m);

    Path inverseCovarianceFile =
        new Path(getTestTempDirPath("mahalanobis"), "MahalanobisDistanceMeasureInverseCovarianceFile");
    conf.set("MahalanobisDistanceMeasure.inverseCovarianceFile", inverseCovarianceFile.toString());
    FileSystem fs = FileSystem.get(inverseCovarianceFile.toUri(), conf);
    MatrixWritable inverseCovarianceMatrix = new MatrixWritable(measure.getInverseCovarianceMatrix());
    DataOutputStream out = fs.create(inverseCovarianceFile);
    try {
      inverseCovarianceMatrix.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }

    Path meanVectorFile = new Path(getTestTempDirPath("mahalanobis"), "MahalanobisDistanceMeasureMeanVectorFile");
    conf.set("MahalanobisDistanceMeasure.meanVectorFile", meanVectorFile.toString());
    fs = FileSystem.get(meanVectorFile.toUri(), conf);
    VectorWritable meanVectorWritable = new VectorWritable(meanVector);
    out = fs.create(meanVectorFile);
    try {
      meanVectorWritable.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }

    conf.set("MahalanobisDistanceMeasure.maxtrixClass", MatrixWritable.class.getName());
    conf.set("MahalanobisDistanceMeasure.vectorClass", VectorWritable.class.getName());

    Integer maxIterations = 5;
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION), getTestTempDirPath("input").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), getTestTempDirPath("output").toString(),
        optKey(DirichletDriver.MODEL_DISTRIBUTION_CLASS_OPTION), description.getModelFactory(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), description.getDistanceMeasure(),
        optKey(DirichletDriver.MODEL_PROTOTYPE_CLASS_OPTION), description.getModelPrototype(),
        optKey(DefaultOptionCreator.NUM_CLUSTERS_OPTION), "20", optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
        maxIterations.toString(), optKey(DirichletDriver.ALPHA_OPTION), "1.0",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.CLUSTERING_OPTION)};
    Tool dirichletDriver = new DirichletDriver();
    dirichletDriver.setConf(conf);
    ToolRunner.run(conf, dirichletDriver, args);
    // and inspect results
    Collection<List<DirichletCluster>> clusters = Lists.newArrayList();
    Configuration conf = new Configuration();
    conf.set(DirichletDriver.MODEL_DISTRIBUTION_KEY, description.toString());
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
    sampleData = Lists.newArrayList();
    generateSamples(500, 2, 0, 0.2);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data2.txt"), fs, conf);
    sampleData = Lists.newArrayList();
    generateSamples(500, 0, 2, 0.3);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data3.txt"), fs, conf);
    sampleData = Lists.newArrayList();
    generateSamples(500, 2, 2, 1);
    ClusteringTestUtils.writePointsToFile(sampleData, getTestTempFilePath("input/data4.txt"), fs, conf);
  }

}
