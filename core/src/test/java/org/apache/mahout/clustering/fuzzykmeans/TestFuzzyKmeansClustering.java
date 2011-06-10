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

package org.apache.mahout.clustering.fuzzykmeans;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestFuzzyKmeansClustering extends MahoutTestCase {

  private FileSystem fs;
  private final DistanceMeasure measure = new EuclideanDistanceMeasure();

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }

  private static double round(double val, int places) {
    long factor = (long) Math.pow(10, places);

    // Shift the decimal the correct number of places
    // to the right.
    val *= factor;

    // Round to the nearest integer.
    long tmp = Math.round(val);

    // Shift the decimal the correct number of places
    // back to the left.
    return (double) tmp / factor;
  }

  private static Vector tweakValue(Vector point) {
    return point.plus(0.1);
  }

  private static void computeCluster(Iterable<Vector> points,
                                     List<SoftCluster> clusterList,
                                     FuzzyKMeansClusterer clusterer,
                                     Map<Integer, List<WeightedVectorWritable>> pointClusterInfo) {

    for (Vector point : points) {
      // calculate point distances for all clusters    
      List<Double> clusterDistanceList = Lists.newArrayList();
      for (SoftCluster cluster : clusterList) {
        clusterDistanceList.add(clusterer.getMeasure().distance(cluster.getCenter(), point));
      }
      // calculate point pdf for all clusters
      List<Double> clusterPdfList = Lists.newArrayList();
      for (int i = 0; i < clusterList.size(); i++) {
        double probWeight = clusterer.computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
        clusterPdfList.add(probWeight);
      }
      // for now just emit the most likely cluster
      int clusterId = -1;
      double clusterPdf = 0;
      for (int i = 0; i < clusterList.size(); i++) {
        // System.out.println("cluster-" + clusters.get(i).getId() + "@ " + ClusterBase.formatVector(center, null));
        double pdf = clusterPdfList.get(i);
        if (pdf > clusterPdf) {
          clusterId = clusterList.get(i).getId();
          clusterPdf = pdf;
        }
      }
      List<WeightedVectorWritable> list = pointClusterInfo.get(clusterId);
      if (list == null) {
        list = Lists.newArrayList();
        pointClusterInfo.put(clusterId, list);
      }
      list.add(new WeightedVectorWritable(clusterPdf, point));
      double totalProb = 0;
      for (int i = 0; i < clusterList.size(); i++) {
        //SoftCluster cluster = clusterList.get(i);
        double probWeight = clusterer.computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
        totalProb += probWeight;
      }
      assertTrue("total probability", Math.abs(1.0 - totalProb) < 0.0001);
    }

    for (SoftCluster cluster : clusterList) {
      System.out.println(cluster.asFormatString(null));
      List<WeightedVectorWritable> list = pointClusterInfo.get(cluster.getId());
      if (list != null) {
        for (WeightedVectorWritable vector : list) {
          System.out.println("\t" + vector);
        }
      }
    }
  }

  @Test
  public void testReferenceImplementation() throws Exception {
    List<Vector> points = TestKmeansClustering.getPoints(TestKmeansClustering.REFERENCE);
    EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();
    for (int k = 0; k < points.size(); k++) {
      System.out.println("test k= " + k);

      List<SoftCluster> clusterList = Lists.newArrayList();
      // pick k initial cluster centers at random
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));
        SoftCluster cluster = new SoftCluster(vec, i, measure);
        // add the center so the centroid will be correct upon output
        //cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }
      Map<Integer, List<WeightedVectorWritable>> pointClusterInfo = Maps.newHashMap();
      // run reference FuzzyKmeans algorithm
      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(points,
                                                                            clusterList,
                                                                            measure,
                                                                            0.001,
                                                                            2,
                                                                            2);
      computeCluster(points,
                     clusters.get(clusters.size() - 1),
                     new FuzzyKMeansClusterer(measure, 0.001, 2),
                     pointClusterInfo);

      // iterate for each cluster
      int size = 0;
      for (List<WeightedVectorWritable> pts : pointClusterInfo.values()) {
        size += pts.size();
      }
      assertEquals("total size", size, points.size());
    }
  }

  @Test
  public void testFuzzyKMeansSeqJob() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersPath, "part-00000"),
                                                           Text.class,
                                                           SoftCluster.class);
      try {
        for (int i = 0; i < k + 1; i++) {
          Vector vec = tweakValue(points.get(i).get());
          SoftCluster cluster = new SoftCluster(vec, i, measure);
          /* add the center so the centroid will be correct upon output */
          cluster.observe(cluster.getCenter(), 1);
          // writer.write(cluster.getIdentifier() + '\t' + SoftCluster.formatCluster(cluster) + '\n');
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        Closeables.closeQuietly(writer);
      }

      // now run the Job using the run() command line options.
      Path output = getTestTempDirPath("output");
      /*      FuzzyKMeansDriver.runJob(pointsPath,
                                     clustersPath,
                                     output,
                                     EuclideanDistanceMeasure.class.getName(),
                                     0.001,
                                     2,
                                     k + 1,
                                     2,
                                     false,
                                     true,
                                     0);
      */
      String[] args = {
          optKey(DefaultOptionCreator.INPUT_OPTION), pointsPath.toString(),
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION),
          clustersPath.toString(),
          optKey(DefaultOptionCreator.OUTPUT_OPTION),
          output.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
          EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION),
          "0.001",
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
          "2",
          optKey(FuzzyKMeansDriver.M_OPTION),
          "2.0",
          optKey(DefaultOptionCreator.CLUSTERING_OPTION),
          optKey(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION),
          optKey(DefaultOptionCreator.OVERWRITE_OPTION),
          optKey(DefaultOptionCreator.METHOD_OPTION),
          DefaultOptionCreator.SEQUENTIAL_METHOD
      };
      new FuzzyKMeansDriver().run(args);
      long count = HadoopUtil.countRecords(new Path(output, "clusteredPoints/part-m-0"), conf);
      assertTrue(count > 0);
    }

  }

  @Test
  public void testFuzzyKMeansMRJob() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersPath, "part-00000"),
                                                           Text.class,
                                                           SoftCluster.class);
      try {
        for (int i = 0; i < k + 1; i++) {
          Vector vec = tweakValue(points.get(i).get());

          SoftCluster cluster = new SoftCluster(vec, i, measure);
          /* add the center so the centroid will be correct upon output */
          cluster.observe(cluster.getCenter(), 1);
          // writer.write(cluster.getIdentifier() + '\t' + SoftCluster.formatCluster(cluster) + '\n');
          writer.append(new Text(cluster.getIdentifier()), cluster);

        }
      } finally {
        Closeables.closeQuietly(writer);
      }

      // now run the Job using the run() command line options.
      Path output = getTestTempDirPath("output");
      /*      FuzzyKMeansDriver.runJob(pointsPath,
                                     clustersPath,
                                     output,
                                     EuclideanDistanceMeasure.class.getName(),
                                     0.001,
                                     2,
                                     k + 1,
                                     2,
                                     false,
                                     true,
                                     0);
      */
      String[] args = {
          optKey(DefaultOptionCreator.INPUT_OPTION),
          pointsPath.toString(),
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION),
          clustersPath.toString(),
          optKey(DefaultOptionCreator.OUTPUT_OPTION),
          output.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
          EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION),
          "0.001",
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION),
          "2",
          optKey(FuzzyKMeansDriver.M_OPTION),
          "2.0",
          optKey(DefaultOptionCreator.CLUSTERING_OPTION),
          optKey(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION),
          optKey(DefaultOptionCreator.OVERWRITE_OPTION)
      };
      ToolRunner.run(new Configuration(), new FuzzyKMeansDriver(), args);
      long count = HadoopUtil.countRecords(new Path(output, "clusteredPoints/part-m-00000"), conf);
      assertTrue(count > 0);
    }

  }

  @Test
  public void testFuzzyKMeansMapper() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Collection<SoftCluster> clusterList = Lists.newArrayList();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i, measure);
        cluster.observe(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);
      DistanceMeasure measure = new EuclideanDistanceMeasure();
      Configuration conf = new Configuration();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");

      DummyRecordWriter<Text, ClusterObservations> mapWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Mapper<WritableComparable<?>, VectorWritable, Text, ClusterObservations>.Context mapContext = DummyRecordWriter
          .build(mapper, conf, mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // now verify mapper output
      assertEquals("Mapper Keys", k + 1, mapWriter.getData().size());

      Map<Vector, Double> pointTotalProbMap = Maps.newHashMap();

      for (Text key : mapWriter.getKeys()) {
        // SoftCluster cluster = SoftCluster.decodeCluster(key);
        List<ClusterObservations> values = mapWriter.getValue(key);

        for (ClusterObservations value : values) {
          Double val = pointTotalProbMap.get(value.getS1());
          double probVal = 0.0;
          if (val != null) {
            probVal = val;
          }
          pointTotalProbMap.put(value.getS1(), probVal + value.getS0());
        }
      }
      for (Map.Entry<Vector, Double> entry : pointTotalProbMap.entrySet()) {
        Vector key = entry.getKey();
        double value = round(entry.getValue(), 1);

        assertEquals("total Prob for Point:" + key, 1.0, value, EPSILON);
      }
    }
  }

  @Test
  public void testFuzzyKMeansCombiner() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Collection<SoftCluster> clusterList = Lists.newArrayList();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i, measure);
        cluster.observe(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      Configuration conf = new Configuration();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY,
          "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");

      DummyRecordWriter<Text, ClusterObservations> mapWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Mapper<WritableComparable<?>, VectorWritable, Text, ClusterObservations>.Context mapContext =
          DummyRecordWriter.build(mapper, conf, mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, ClusterObservations> combinerWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Reducer<Text, ClusterObservations, Text, ClusterObservations>.Context combinerContext =
          DummyRecordWriter.build(combiner, conf, combinerWriter, Text.class, ClusterObservations.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<ClusterObservations> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // now verify the combiner output
      assertEquals("Combiner Output", k + 1, combinerWriter.getData().size());

      for (Text key : combinerWriter.getKeys()) {
        List<ClusterObservations> values = combinerWriter.getValue(key);
        assertEquals("too many values", 1, values.size());
      }
    }
  }

  @Test
  public void testFuzzyKMeansReducer() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Collection<SoftCluster> clusterList = Lists.newArrayList();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i, measure);
        // cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);
      DistanceMeasure measure = new EuclideanDistanceMeasure();
      Configuration conf = new Configuration();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");

      DummyRecordWriter<Text, ClusterObservations> mapWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Mapper<WritableComparable<?>, VectorWritable, Text, ClusterObservations>.Context mapContext =
          DummyRecordWriter.build(mapper, conf, mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, ClusterObservations> combinerWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Reducer<Text, ClusterObservations, Text, ClusterObservations>.Context combinerContext =
          DummyRecordWriter.build(combiner, conf, combinerWriter, Text.class, ClusterObservations.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<ClusterObservations> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // run reducer
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      DummyRecordWriter<Text, SoftCluster> reducerWriter = new DummyRecordWriter<Text, SoftCluster>();
      Reducer<Text, ClusterObservations, Text, SoftCluster>.Context reducerContext =
          DummyRecordWriter.build(reducer, conf, reducerWriter, Text.class, ClusterObservations.class);
      reducer.setup(clusterList, conf);

      for (Text key : combinerWriter.getKeys()) {
        List<ClusterObservations> values = combinerWriter.getValue(key);
        reducer.reduce(new Text(key), values, reducerContext);
      }

      // now verify the reducer output
      assertEquals("Reducer Output", k + 1, combinerWriter.getData().size());

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = Lists.newArrayList();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());
        reference.add(new SoftCluster(vec, i, measure));
      }
      Collection<Vector> pointsVectors = Lists.newArrayList();
      for (VectorWritable point : points) {
        pointsVectors.add(point.get());
      }

      FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, 0.001, 2);
      FuzzyKMeansClusterer.runFuzzyKMeansIteration(pointsVectors, reference, clusterer);

      for (SoftCluster key : reference) {
        String clusterId = key.getIdentifier();
        List<SoftCluster> values = reducerWriter.getValue(new Text(clusterId));
        SoftCluster cluster = values.get(0);
        System.out.println("ref= " + key.toString() + " cluster= " + cluster);
        cluster.computeParameters();
        assertEquals("key center: " + AbstractCluster.formatVector(key.getCenter(), null) + " does not equal cluster: "
            + AbstractCluster.formatVector(cluster.getCenter(), null), key.getCenter(), cluster.getCenter());
      }
    }
  }

  @Test
  public void testFuzzyKMeansClusterMapper() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.REFERENCE);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Collection<SoftCluster> clusterList = Lists.newArrayList();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i, measure);
        cluster.observe(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }
      for (SoftCluster softCluster : clusterList) {
        softCluster.computeParameters();
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);
      DistanceMeasure measure = new EuclideanDistanceMeasure();

      Configuration conf = new Configuration();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");

      DummyRecordWriter<Text, ClusterObservations> mapWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Mapper<WritableComparable<?>, VectorWritable, Text, ClusterObservations>.Context mapContext =
          DummyRecordWriter.build(mapper, conf, mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, ClusterObservations> combinerWriter = new DummyRecordWriter<Text, ClusterObservations>();
      Reducer<Text, ClusterObservations, Text, ClusterObservations>.Context combinerContext =
          DummyRecordWriter.build(combiner, conf, combinerWriter, Text.class, ClusterObservations.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<ClusterObservations> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // run reducer
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      DummyRecordWriter<Text, SoftCluster> reducerWriter = new DummyRecordWriter<Text, SoftCluster>();
      Reducer<Text, ClusterObservations, Text, SoftCluster>.Context reducerContext =
          DummyRecordWriter.build(reducer, conf, reducerWriter, Text.class, ClusterObservations.class);
      reducer.setup(clusterList, conf);

      for (Text key : combinerWriter.getKeys()) {
        List<ClusterObservations> values = combinerWriter.getValue(key);
        reducer.reduce(new Text(key), values, reducerContext);
      }

      // run clusterMapper
      Collection<SoftCluster> reducerClusters = Lists.newArrayList();
      for (Text key : reducerWriter.getKeys()) {
        List<SoftCluster> values = reducerWriter.getValue(key);
        reducerClusters.add(values.get(0));
      }
      for (SoftCluster softCluster : reducerClusters) {
        softCluster.computeParameters();
      }

      FuzzyKMeansClusterMapper clusterMapper = new FuzzyKMeansClusterMapper();
      DummyRecordWriter<IntWritable, WeightedVectorWritable> clusterWriter =
          new DummyRecordWriter<IntWritable, WeightedVectorWritable>();
      Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable>.Context clusterContext =
          DummyRecordWriter.build(clusterMapper, conf, clusterWriter);
      clusterMapper.setup(reducerClusters, conf);

      for (VectorWritable point : points) {
        clusterMapper.map(new Text(), point, clusterContext);
      }

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = Lists.newArrayList();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());
        reference.add(new SoftCluster(vec, i, measure));
      }
      Map<Integer, List<WeightedVectorWritable>> refClusters = Maps.newHashMap();
      Collection<Vector> pointsVectors = Lists.newArrayList();
      for (VectorWritable point : points) {
        pointsVectors.add(point.get());
      }

      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(pointsVectors,
                                                                            reference,
                                                                            new EuclideanDistanceMeasure(),
                                                                            0.001,
                                                                            2,
                                                                            1);

      computeCluster(pointsVectors, clusters.get(clusters.size() - 1),
                     new FuzzyKMeansClusterer(new EuclideanDistanceMeasure(), 0.001, 2), refClusters);

      // Now compare the clustermapper results with reference implementation
      assertEquals("mapper and reference sizes", refClusters.size(), clusterWriter.getKeys().size());
      for (Map.Entry<Integer, List<WeightedVectorWritable>> entry : refClusters.entrySet()) {
        int key = entry.getKey();
        List<WeightedVectorWritable> value = entry.getValue();
        System.out.println("refClusters=" + value + " mapClusters=" + clusterWriter.getValue(new IntWritable(key)));
        assertEquals("cluster " + key + " sizes", value.size(), clusterWriter.getValue(new IntWritable(key)).size());
      }
      // make sure all points are allocated to a cluster
      int size = 0;
      for (List<WeightedVectorWritable> pts : refClusters.values()) {
        size += pts.size();
      }
      assertEquals("total size", size, points.size());
    }
  }

  @Test
  public void testClusterObservationsSerialization() throws Exception {
    double[] data = { 1.1, 2.2, 3.3 };
    Vector vector = new DenseVector(data);
    ClusterObservations reference = new ClusterObservations(1, 2.0, vector, vector);
    DataOutputBuffer out = new DataOutputBuffer();
    reference.write(out);
    ClusterObservations info = new ClusterObservations();
    DataInputBuffer in = new DataInputBuffer();
    in.reset(out.getData(), out.getLength());
    info.readFields(in);
    assertEquals("probability", reference.getS0(), info.getS0(), EPSILON);
    assertEquals("point total", reference.getS1(), info.getS1());
    assertEquals("combiner", reference.getCombinerState(), info.getCombinerState());
  }

}
