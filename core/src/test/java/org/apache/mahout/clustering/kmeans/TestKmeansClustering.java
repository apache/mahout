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

package org.apache.mahout.clustering.kmeans;

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
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TestKmeansClustering extends MahoutTestCase {

  public static final double[][] reference = { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 },
      { 5, 5 } };

  private static final int[][] expectedNumPoints = { { 9 }, { 4, 5 }, { 4, 4, 1 }, { 1, 2, 1, 5 }, { 1, 1, 1, 2, 4 },
      { 1, 1, 1, 1, 1, 4 }, { 1, 1, 1, 1, 1, 2, 2 }, { 1, 1, 1, 1, 1, 1, 2, 1 }, { 1, 1, 1, 1, 1, 1, 1, 1, 1 } };

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

  public static List<VectorWritable> getPointsWritable(double[][] raw) {
    List<VectorWritable> points = new ArrayList<VectorWritable>();
    for (double[] fr : raw) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  public static List<Vector> getPoints(double[][] raw) {
    List<Vector> points = new ArrayList<Vector>();
    for (double[] fr : raw) {
      Vector vec = new SequentialAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  /** Story: Test the reference implementation */
  public void testReferenceImplementation() throws Exception {
    List<Vector> points = getPoints(reference);
    DistanceMeasure measure = new EuclideanDistanceMeasure();
    // try all possible values of k
    for (int k = 0; k < points.size(); k++) {
      System.out.println("Test k=" + (k + 1) + ':');
      // pick k initial cluster centers at random
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i);
        clusters.add(new VisibleCluster(vec));
      }
      // iterate clusters until they converge
      int maxIter = 10;
      List<List<Cluster>> clustersList = KMeansClusterer.clusterPoints(points, clusters, measure, maxIter, 0.001);
      clusters = clustersList.get(clustersList.size() - 1);
      for (int c = 0; c < clusters.size(); c++) {
        Cluster cluster = clusters.get(c);
        System.out.println(cluster.toString());
        assertEquals("Cluster " + c + " test " + (k + 1), expectedNumPoints[k][c], cluster.getNumPoints());
      }
    }
  }

  public void testStd() {
    List<Vector> points = getPoints(reference);
    Cluster c = new Cluster(points.get(0));
    for (Vector p : points) {
      c.addPoint(p);
      if (c.getNumPoints() > 1) {
        assertTrue(c.getStd() > 0.0);
      }
    }
  }

  private static Map<String, Cluster> loadClusterMap(List<Cluster> clusters) {
    Map<String, Cluster> clusterMap = new HashMap<String, Cluster>();

    for (Cluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    return clusterMap;
  }

  /** Story: test that the mapper will map input points to the nearest cluster */
  public void testKMeansMapper() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, "");
    List<VectorWritable> points = getPointsWritable(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyRecordWriter<Text, KMeansInfo> mapWriter = new DummyRecordWriter<Text, KMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, KMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                   conf,
                                                                                                                   mapWriter);
      List<Cluster> clusters = new ArrayList<Cluster>();

      for (int i = 0; i < k + 1; i++) {
        Cluster cluster = new Cluster(points.get(i).get(), i);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.setup(clusters, measure);

      // map the data
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }
      assertEquals("Number of map results", k + 1, mapWriter.getData().size());
      // now verify that all points are correctly allocated
      EuclideanDistanceMeasure euclideanDistanceMeasure = measure;
      Map<String, Cluster> clusterMap = loadClusterMap(clusters);
      for (Text key : mapWriter.getKeys()) {
        Cluster cluster = clusterMap.get(key.toString());
        List<KMeansInfo> values = mapWriter.getValue(key);
        for (KMeansInfo value : values) {
          double distance = euclideanDistanceMeasure.distance(cluster.getCenter(), value.getPointTotal());
          for (Cluster c : clusters) {
            assertTrue("distance error", distance <= euclideanDistanceMeasure.distance(value.getPointTotal(), c.getCenter()));
          }
        }
      }
    }
  }

  /**
   * Story: test that the combiner will produce partial cluster totals for all of the clusters and points that
   * it sees
   */
  public void testKMeansCombiner() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, "");
    List<VectorWritable> points = getPointsWritable(reference);
    for (int k = 0; k < points.size(); k++) {
      // pick k initial cluster centers at random
      DummyRecordWriter<Text, KMeansInfo> mapWriter = new DummyRecordWriter<Text, KMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, KMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                   conf,
                                                                                                                   mapWriter);
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i).get();

        Cluster cluster = new Cluster(vec, i);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.setup(clusters, measure);
      // map the data
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }
      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyRecordWriter<Text, KMeansInfo> combinerWriter = new DummyRecordWriter<Text, KMeansInfo>();
      Reducer<Text, KMeansInfo, Text, KMeansInfo>.Context combinerContext = DummyRecordWriter.build(combiner,
                                                                                                    conf,
                                                                                                    combinerWriter,
                                                                                                    Text.class,
                                                                                                    KMeansInfo.class);
      for (Text key : mapWriter.getKeys()) {
        combiner.reduce(new Text(key), mapWriter.getValue(key), combinerContext);
      }

      assertEquals("Number of map results", k + 1, combinerWriter.getData().size());
      // now verify that all points are accounted for
      int count = 0;
      Vector total = new DenseVector(2);
      for (Text key : combinerWriter.getKeys()) {
        List<KMeansInfo> values = combinerWriter.getValue(key);
        assertEquals("too many values", 1, values.size());
        // String value = values.get(0).toString();
        KMeansInfo info = values.get(0);

        count += info.getPoints();
        total = total.plus(info.getPointTotal());
      }
      assertEquals("total points", 9, count);
      assertEquals("point total[0]", 27, (int) total.get(0));
      assertEquals("point total[1]", 27, (int) total.get(1));
    }
  }

  /**
   * Story: test that the reducer will sum the partial cluster totals for all of the clusters and points that
   * it sees
   */
  public void testKMeansReducer() throws Exception {
    KMeansMapper mapper = new KMeansMapper();
    EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, "");
    List<VectorWritable> points = getPointsWritable(reference);
    for (int k = 0; k < points.size(); k++) {
      System.out.println("K = " + k);
      // pick k initial cluster centers at random
      DummyRecordWriter<Text, KMeansInfo> mapWriter = new DummyRecordWriter<Text, KMeansInfo>();
      Mapper<WritableComparable<?>, VectorWritable, Text, KMeansInfo>.Context mapContext = DummyRecordWriter.build(mapper,
                                                                                                                   conf,
                                                                                                                   mapWriter);
      List<Cluster> clusters = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i).get();
        Cluster cluster = new Cluster(vec, i);
        // add the center so the centroid will be correct upon output
        // cluster.addPoint(cluster.getCenter());
        clusters.add(cluster);
      }
      mapper.setup(clusters, new EuclideanDistanceMeasure());
      // map the data
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapContext);
      }
      // now combine the data
      KMeansCombiner combiner = new KMeansCombiner();
      DummyRecordWriter<Text, KMeansInfo> combinerWriter = new DummyRecordWriter<Text, KMeansInfo>();
      Reducer<Text, KMeansInfo, Text, KMeansInfo>.Context combinerContext = DummyRecordWriter.build(combiner,
                                                                                                    conf,
                                                                                                    combinerWriter,
                                                                                                    Text.class,
                                                                                                    KMeansInfo.class);
      for (Text key : mapWriter.getKeys()) {
        combiner.reduce(new Text(key), mapWriter.getValue(key), combinerContext);
      }

      // now reduce the data
      KMeansReducer reducer = new KMeansReducer();
      reducer.setup(clusters, measure);
      DummyRecordWriter<Text, Cluster> reducerWriter = new DummyRecordWriter<Text, Cluster>();
      Reducer<Text, KMeansInfo, Text, Cluster>.Context reducerContext = DummyRecordWriter.build(reducer,
                                                                                                conf,
                                                                                                reducerWriter,
                                                                                                Text.class,
                                                                                                KMeansInfo.class);
      for (Text key : combinerWriter.getKeys()) {
        reducer.reduce(new Text(key), combinerWriter.getValue(key), reducerContext);
      }

      assertEquals("Number of map results", k + 1, reducerWriter.getData().size());

      // compute the reference result after one iteration and compare
      List<Cluster> reference = new ArrayList<Cluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i).get();
        reference.add(new Cluster(vec, i));
      }
      List<Vector> pointsVectors = new ArrayList<Vector>();
      for (VectorWritable point : points) {
        pointsVectors.add(point.get());
      }
      boolean converged = KMeansClusterer.runKMeansIteration(pointsVectors, reference, measure, 0.001);
      if (k == 8) {
        assertTrue("not converged? " + k, converged);
      } else {
        assertFalse("converged? " + k, converged);
      }

      // now verify that all clusters have correct centers
      converged = true;
      for (Cluster ref : reference) {
        String key = ref.getIdentifier();
        List<Cluster> values = reducerWriter.getValue(new Text(key));
        Cluster cluster = values.get(0);
        converged = converged && cluster.isConverged();
        // Since we aren't roundtripping through Writable, we need to compare the reference center with the
        // cluster centroid
        cluster.recomputeCenter();
        assertEquals(ref.getCenter(), cluster.getCenter());
      }
      if (k == 8) {
        assertTrue("not converged? " + k, converged);
      } else {
        assertFalse("converged? " + k, converged);
      }
    }
  }

  /** Story: User wishes to run kmeans job on reference data */
  public void testKMeansMRJob() throws Exception {
    List<VectorWritable> points = getPointsWritable(reference);

    Path pointsPath = getTestTempDirPath("points");
    Path clustersPath = getTestTempDirPath("clusters");
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file2"), fs, conf);
    for (int k = 1; k < points.size(); k++) {
      System.out.println("testKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      Path path = new Path(clustersPath, "part-00000");
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Cluster.class);

      for (int i = 0; i < k + 1; i++) {
        Vector vec = points.get(i).get();

        Cluster cluster = new Cluster(vec, i);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter());
        writer.append(new Text(cluster.getIdentifier()), cluster);
      }
      writer.close();
      // now run the Job
      Path outputPath = getTestTempDirPath("output");
      //KMeansDriver.runJob(pointsPath, clustersPath, outputPath, EuclideanDistanceMeasure.class.getName(), 0.001, 10, k + 1, true);
      String[] args = { 
          optKey(DefaultOptionCreator.INPUT_OPTION), pointsPath.toString(), 
          optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION), clustersPath.toString(), 
          optKey(DefaultOptionCreator.OUTPUT_OPTION), outputPath.toString(),
          optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION), EuclideanDistanceMeasure.class.getName(),
          optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.001", 
          optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "2",
          optKey(DefaultOptionCreator.CLUSTERING_OPTION), 
          optKey(DefaultOptionCreator.OVERWRITE_OPTION) };
      new KMeansDriver().run(args);

      // now compare the expected clusters with actual
      Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
      // assertEquals("output dir files?", 4, outFiles.length);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(clusteredPointsPath, "part-m-00000"), conf);
      int[] expect = expectedNumPoints[k];
      DummyOutputCollector<IntWritable, WeightedVectorWritable> collector = new DummyOutputCollector<IntWritable, WeightedVectorWritable>();
      // The key is the clusterId
      IntWritable clusterId = new IntWritable(0);
      // The value is the weighted vector
      WeightedVectorWritable value = new WeightedVectorWritable();
      while (reader.next(clusterId, value)) {
        collector.collect(clusterId, value);
        clusterId = new IntWritable(0);
        value = new WeightedVectorWritable();
      }
      reader.close();
      if (k == 2) {
        // cluster 3 is empty so won't appear in output
        assertEquals("clusters[" + k + ']', expect.length - 1, collector.getKeys().size());
      } else {
        assertEquals("clusters[" + k + ']', expect.length, collector.getKeys().size());
      }
    }
  }

  /** Story: User wants to use canopy clustering to input the initial clusters for kmeans job. */
  public void testKMeansWithCanopyClusterInput() throws Exception {
    List<VectorWritable> points = getPointsWritable(reference);

    Path pointsPath = getTestTempDirPath("points");
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points, new Path(pointsPath, "file2"), fs, conf);

    Path outputPath = getTestTempDirPath("output");
    // now run the Canopy job
    CanopyDriver.runJob(pointsPath, outputPath, ManhattanDistanceMeasure.class.getName(), 3.1, 2.1, false);

    // now run the KMeans job
    KMeansDriver.runJob(pointsPath,
                        new Path(outputPath, "clusters-0"),
                        outputPath,
                        EuclideanDistanceMeasure.class.getName(),
                        0.001,
                        10,
                        1,
                        true);

    // now compare the expected clusters with actual
    Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
    //String[] outFiles = outDir.list();
    //assertEquals("output dir files?", 4, outFiles.length);
    DummyOutputCollector<IntWritable, WeightedVectorWritable> collector = new DummyOutputCollector<IntWritable, WeightedVectorWritable>();
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(clusteredPointsPath, "part-m-00000"), conf);

    // The key is the clusterId
    IntWritable clusterId = new IntWritable(0);
    // The value is the vector
    WeightedVectorWritable value = new WeightedVectorWritable();
    while (reader.next(clusterId, value)) {
      collector.collect(clusterId, value);
      clusterId = new IntWritable(0);
      value = new WeightedVectorWritable();

    }
    reader.close();

    assertEquals("num points[0]", 4, collector.getValue(new IntWritable(0)).size());
    assertEquals("num points[1]", 5, collector.getValue(new IntWritable(1)).size());
  }
}
