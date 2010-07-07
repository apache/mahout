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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestFuzzyKmeansClustering extends MahoutTestCase {

  private FileSystem fs;

  private static void rmr(String path) {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (String content : contents) {
          rmr(f.toString() + File.separator + content);
        }
      }
      f.delete();
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rmr("output");
    rmr("testdata");
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

  private static void computeCluster(List<Vector> points,
                                     List<SoftCluster> clusterList,
                                     FuzzyKMeansClusterer clusterer,
                                     Map<Integer, List<WeightedVectorWritable>> pointClusterInfo) {

    for (Vector point : points) {
      // calculate point distances for all clusters    
      List<Double> clusterDistanceList = new ArrayList<Double>();
      for (SoftCluster cluster : clusterList) {
        clusterDistanceList.add(clusterer.getMeasure().distance(cluster.getCenter(), point));
      }
      // calculate point pdf for all clusters
      List<Double> clusterPdfList = new ArrayList<Double>();
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
        list = new ArrayList<WeightedVectorWritable>();
        pointClusterInfo.put(clusterId, list);
      }
      list.add(new WeightedVectorWritable(clusterPdf, new VectorWritable(point)));
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

  public void testReferenceImplementation() throws Exception {
    List<Vector> points = TestKmeansClustering.getPoints(TestKmeansClustering.reference);
    for (int k = 0; k < points.size(); k++) {
      System.out.println("test k= " + k);

      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();
      // pick k initial cluster centers at random
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));
        SoftCluster cluster = new SoftCluster(vec, i);
        // add the center so the centroid will be correct upon output
        //cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }
      Map<Integer, List<WeightedVectorWritable>> pointClusterInfo = new HashMap<Integer, List<WeightedVectorWritable>>();
      // run reference FuzzyKmeans algorithm
      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(points,
                                                                            clusterList,
                                                                            new EuclideanDistanceMeasure(),
                                                                            0.001,
                                                                            2,
                                                                            2);
      computeCluster(points,
                     clusters.get(clusters.size() - 1),
                     new FuzzyKMeansClusterer(new EuclideanDistanceMeasure(), 0.001, 2),
                     pointClusterInfo);

      // iterate for each cluster
      int size = 0;
      for (List<WeightedVectorWritable> pts : pointClusterInfo.values()) {
        size += pts.size();
      }
      assertEquals("total size", size, points.size());
    }
  }

  public void testFuzzyKMeansMRJob() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);

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
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter(), 1);
        /*
         * writer.write(cluster.getIdentifier() + '\t' + SoftCluster.formatCluster(cluster) + '\n');
         */
        writer.append(new Text(cluster.getIdentifier()), cluster);

      }
      writer.close();

      // now run the Job
      Path output = getTestTempDirPath("output");
      FuzzyKMeansDriver.runJob(pointsPath,
                               clustersPath,
                               output,
                               EuclideanDistanceMeasure.class.getName(),
                               0.001,
                               2,
                               1,
                               k + 1,
                               2,
                               false,
                               true,
                               0);

      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(output, "clusteredPoints/part-m-00000"), conf);
      IntWritable key = new IntWritable();
      WeightedVectorWritable out = new WeightedVectorWritable();
      while (reader.next(key, out)) {
        // make sure we can read all the clusters
      }
      reader.close();

    }

  }

  public void testFuzzyKMeansMapper() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i);
        cluster.addPoint(cluster.getCenter(), 1);
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

      DummyRecordWriter<Text, FuzzyKMeansInfo> mapWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, FuzzyKMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                        conf,
                                                                                                                        mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // now verify mapper output
      assertEquals("Mapper Keys", k + 1, mapWriter.getData().size());

      Map<Vector, Double> pointTotalProbMap = new HashMap<Vector, Double>();

      for (Text key : mapWriter.getKeys()) {
        // SoftCluster cluster = SoftCluster.decodeCluster(key);
        List<FuzzyKMeansInfo> values = mapWriter.getValue(key);

        for (FuzzyKMeansInfo value : values) {
          Double val = pointTotalProbMap.get(value.getVector());
          double probVal = 0.0;
          if (val != null) {
            probVal = val;
          }
          pointTotalProbMap.put(value.getVector(), probVal + value.getProbability());
        }
      }
      for (Map.Entry<Vector, Double> entry : pointTotalProbMap.entrySet()) {
        Vector key = entry.getKey();
        double value = round(entry.getValue(), 1);

        assertEquals("total Prob for Point:" + key, 1.0, value);
      }
    }
  }

  public void testFuzzyKMeansCombiner() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i);
        cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      Configuration conf = new Configuration();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");

      DummyRecordWriter<Text, FuzzyKMeansInfo> mapWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, FuzzyKMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                        conf,
                                                                                                                        mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, FuzzyKMeansInfo> combinerWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Reducer<Text, FuzzyKMeansInfo, Text, FuzzyKMeansInfo>.Context combinerContext = DummyRecordWriter
          .build(combiner, conf, combinerWriter, Text.class, FuzzyKMeansInfo.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // now verify the combiner output
      assertEquals("Combiner Output", k + 1, combinerWriter.getData().size());

      for (Text key : combinerWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerWriter.getValue(key);
        assertEquals("too many values", 1, values.size());
      }
    }
  }

  public void testFuzzyKMeansReducer() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i);
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

      DummyRecordWriter<Text, FuzzyKMeansInfo> mapWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, FuzzyKMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                        conf,
                                                                                                                        mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, FuzzyKMeansInfo> combinerWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Reducer<Text, FuzzyKMeansInfo, Text, FuzzyKMeansInfo>.Context combinerContext = DummyRecordWriter
          .build(combiner, conf, combinerWriter, Text.class, FuzzyKMeansInfo.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // run reducer
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      DummyRecordWriter<Text, SoftCluster> reducerWriter = new DummyRecordWriter<Text, SoftCluster>();
      Reducer<Text, FuzzyKMeansInfo, Text, SoftCluster>.Context reducerContext = DummyRecordWriter.build(reducer,
                                                                                                         conf,
                                                                                                         reducerWriter,
                                                                                                         Text.class,
                                                                                                         FuzzyKMeansInfo.class);
      reducer.setup(clusterList, conf);

      for (Text key : combinerWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerWriter.getValue(key);
        reducer.reduce(new Text(key), values, reducerContext);
      }

      // now verify the reducer output
      assertEquals("Reducer Output", k + 1, combinerWriter.getData().size());

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = new ArrayList<SoftCluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());
        reference.add(new SoftCluster(vec, i));
      }
      List<Vector> pointsVectors = new ArrayList<Vector>();
      for (VectorWritable point : points) {
        pointsVectors.add(point.get());
      }

      FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, 0.001, 2);
      FuzzyKMeansClusterer.runFuzzyKMeansIteration(pointsVectors, reference, clusterer);

      for (SoftCluster key : reference) {
        String clusterId = key.getIdentifier();
        List<SoftCluster> values = reducerWriter.getValue(new Text(clusterId));
        SoftCluster cluster = values.get(0);
        System.out.println("ref= " + key.toString() + " cluster= " + cluster.toString());
        cluster.recomputeCenter();
        assertEquals("key center: " + key.getCenter().asFormatString() + " does not equal cluster: "
            + cluster.getCenter().asFormatString(), key.getCenter(), cluster.getCenter());
      }
    }
  }

  public void testFuzzyKMeansClusterMapper() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());

        SoftCluster cluster = new SoftCluster(vec, i);
        cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }
      for (SoftCluster softCluster : clusterList) {
        softCluster.recomputeCenter();
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

      DummyRecordWriter<Text, FuzzyKMeansInfo> mapWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, FuzzyKMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                        conf,
                                                                                                                        mapWriter);
      mapper.setup(mapContext);
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }

      // run combiner
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      DummyRecordWriter<Text, FuzzyKMeansInfo> combinerWriter = new DummyRecordWriter<Text, FuzzyKMeansInfo>();
      Reducer<Text, FuzzyKMeansInfo, Text, FuzzyKMeansInfo>.Context combinerContext = DummyRecordWriter
          .build(combiner, conf, combinerWriter, Text.class, FuzzyKMeansInfo.class);
      combiner.setup(combinerContext);
      for (Text key : mapWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = mapWriter.getValue(key);
        combiner.reduce(new Text(key), values, combinerContext);
      }

      // run reducer
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      DummyRecordWriter<Text, SoftCluster> reducerWriter = new DummyRecordWriter<Text, SoftCluster>();
      Reducer<Text, FuzzyKMeansInfo, Text, SoftCluster>.Context reducerContext = DummyRecordWriter.build(reducer,
                                                                                                         conf,
                                                                                                         reducerWriter,
                                                                                                         Text.class,
                                                                                                         FuzzyKMeansInfo.class);
      reducer.setup(clusterList, conf);

      for (Text key : combinerWriter.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerWriter.getValue(key);
        reducer.reduce(new Text(key), values, reducerContext);
      }

      // run clusterMapper
      List<SoftCluster> reducerClusters = new ArrayList<SoftCluster>();
      for (Text key : reducerWriter.getKeys()) {
        List<SoftCluster> values = reducerWriter.getValue(key);
        reducerClusters.add(values.get(0));
      }
      for (SoftCluster softCluster : reducerClusters) {
        softCluster.recomputeCenter();
      }

      FuzzyKMeansClusterMapper clusterMapper = new FuzzyKMeansClusterMapper();
      DummyRecordWriter<IntWritable, WeightedVectorWritable> clusterWriter = new DummyRecordWriter<IntWritable, WeightedVectorWritable>();
      Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable>.Context clusterContext = DummyRecordWriter
          .build(clusterMapper, conf, clusterWriter);
      clusterMapper.setup(reducerClusters, conf);

      for (VectorWritable point : points) {
        clusterMapper.map(new Text(), point, clusterContext);
      }

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = new ArrayList<SoftCluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());
        reference.add(new SoftCluster(vec, i));
      }
      Map<Integer, List<WeightedVectorWritable>> refClusters = new HashMap<Integer, List<WeightedVectorWritable>>();
      List<Vector> pointsVectors = new ArrayList<Vector>();
      for (VectorWritable point : points) {
        pointsVectors.add((Vector) point.get());
      }

      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(pointsVectors,
                                                                            reference,
                                                                            new EuclideanDistanceMeasure(),
                                                                            0.001,
                                                                            2,
                                                                            1);

      computeCluster(pointsVectors, clusters.get(clusters.size() - 1), new FuzzyKMeansClusterer(new EuclideanDistanceMeasure(),
                                                                                                0.001,
                                                                                                2), refClusters);

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

}
