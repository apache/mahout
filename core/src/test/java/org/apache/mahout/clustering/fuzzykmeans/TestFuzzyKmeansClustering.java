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
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.DummyReporter;
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

  private static void computeCluster(List<Vector> points, List<SoftCluster> clusterList, FuzzyKMeansClusterer clusterer,
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
        SoftCluster cluster = clusterList.get(i);
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
      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(points, clusterList, new EuclideanDistanceMeasure(),
          0.001, 2, 2);
      computeCluster(points, clusters.get(clusters.size() - 1), new FuzzyKMeansClusterer(new EuclideanDistanceMeasure(), 0.001, 2),
          pointClusterInfo);

      // iterate for each cluster
      int size = 0;
      for (int cId : pointClusterInfo.keySet()) {
        List<WeightedVectorWritable> pts = pointClusterInfo.get(cId);
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
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(clustersPath, "part-00000"), Text.class,
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
      FuzzyKMeansDriver.runJob(pointsPath, clustersPath, output, EuclideanDistanceMeasure.class.getName(), 0.001,
          2, 1, k + 1, 2, false, true, 0);

      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(output, "clusteredPoints/part-00000"), conf);
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

      JobConf conf = new JobConf();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");
      mapper.configure(conf);

      DummyOutputCollector<Text, FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }

      // now verify mapper output
      assertEquals("Mapper Keys", k + 1, mapCollector.getData().size());

      Map<Vector, Double> pointTotalProbMap = new HashMap<Vector, Double>();

      for (Text key : mapCollector.getKeys()) {
        // SoftCluster cluster = SoftCluster.decodeCluster(key);
        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);

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

      JobConf conf = new JobConf();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");
      mapper.configure(conf);

      DummyOutputCollector<Text, FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }

      // run combiner
      DummyOutputCollector<Text, FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      combiner.configure(conf);

      for (Text key : mapCollector.getKeys()) {

        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector, null);
      }

      // now verify the combiner output
      assertEquals("Combiner Output", k + 1, combinerCollector.getData().size());

      for (Text key : combinerCollector.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerCollector.getValue(key);
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

      JobConf conf = new JobConf();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");
      mapper.configure(conf);

      DummyOutputCollector<Text, FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }

      // run combiner
      DummyOutputCollector<Text, FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      combiner.configure(conf);

      for (Text key : mapCollector.getKeys()) {
        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector, null);
      }

      // run reducer
      DummyOutputCollector<Text, SoftCluster> reducerCollector = new DummyOutputCollector<Text, SoftCluster>();
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      reducer.config(clusterList);
      reducer.configure(conf);

      for (Text key : combinerCollector.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerCollector.getValue(key);
        reducer.reduce(new Text(key), values.iterator(), reducerCollector, new DummyReporter());
      }

      // now verify the reducer output
      assertEquals("Reducer Output", k + 1, combinerCollector.getData().size());

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = new ArrayList<SoftCluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i).get());
        reference.add(new SoftCluster(vec, i));
      }
      List<Vector> pointsVectors = new ArrayList<Vector>();
      for (VectorWritable point : points) {
        pointsVectors.add((Vector) point.get());
      }

      DistanceMeasure measure = new EuclideanDistanceMeasure();
      FuzzyKMeansClusterer clusterer = new FuzzyKMeansClusterer(measure, 0.001, 2);
      FuzzyKMeansClusterer.runFuzzyKMeansIteration(pointsVectors, reference, clusterer);

      for (SoftCluster key : reference) {
        String clusterId = key.getIdentifier();
        List<SoftCluster> values = reducerCollector.getValue(new Text(clusterId));
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

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      JobConf conf = new JobConf();
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      conf.set(FuzzyKMeansConfigKeys.EMIT_MOST_LIKELY_KEY, "true");
      conf.set(FuzzyKMeansConfigKeys.THRESHOLD_KEY, "0");
      mapper.configure(conf);

      DummyOutputCollector<Text, FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }
      for (SoftCluster softCluster : clusterList) {
        softCluster.recomputeCenter();
      }
      // run combiner
      DummyOutputCollector<Text, FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text, FuzzyKMeansInfo>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      combiner.configure(conf);

      for (Text key : mapCollector.getKeys()) {

        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector, null);
      }

      // run reducer
      DummyOutputCollector<Text, SoftCluster> reducerCollector = new DummyOutputCollector<Text, SoftCluster>();
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      reducer.config(clusterList);
      reducer.configure(conf);

      for (Text key : combinerCollector.getKeys()) {
        List<FuzzyKMeansInfo> values = combinerCollector.getValue(key);
        reducer.reduce(new Text(key), values.iterator(), reducerCollector, null);
      }

      // run clusterMapper
      List<SoftCluster> reducerCluster = new ArrayList<SoftCluster>();

      for (Text key : reducerCollector.getKeys()) {
        List<SoftCluster> values = reducerCollector.getValue(key);
        reducerCluster.add(values.get(0));
      }
      for (SoftCluster softCluster : reducerCluster) {
        softCluster.recomputeCenter();
      }

      DummyOutputCollector<IntWritable, WeightedVectorWritable> clusterMapperCollector = new DummyOutputCollector<IntWritable, WeightedVectorWritable>();

      FuzzyKMeansClusterMapper clusterMapper = new FuzzyKMeansClusterMapper();
      clusterMapper.config(reducerCluster);
      clusterMapper.configure(conf);

      for (VectorWritable point : points) {
        clusterMapper.map(new Text(), point, clusterMapperCollector, null);
      }

      // now run for one iteration of referencefuzzykmeans and compare the
      // results
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

      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(pointsVectors, reference,
          new EuclideanDistanceMeasure(), 0.001, 2, 1);

      computeCluster(pointsVectors, clusters.get(clusters.size() - 1), new FuzzyKMeansClusterer(new EuclideanDistanceMeasure(),
          0.001, 2), refClusters);

      // Now compare the clustermapper results with reference implementation
      assertEquals("mapper and reference sizes", refClusters.size(), clusterMapperCollector.getKeys().size());
      for (int pcId : refClusters.keySet()) {
        assertEquals("cluster " + pcId + " sizes", refClusters.get(pcId).size(), clusterMapperCollector.getValue(
            new IntWritable(pcId)).size());
      }
      // make sure all points are allocated to a cluster
      int size = 0;
      for (int cId : refClusters.keySet()) {
        List<WeightedVectorWritable> pts = refClusters.get(cId);
        size += pts.size();
      }
      assertEquals("total size", size, points.size());
    }
  }

}
