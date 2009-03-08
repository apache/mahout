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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.DummyOutputCollector;
import org.apache.mahout.utils.EuclideanDistanceMeasure;

public class TestFuzzyKmeansClustering extends TestCase {

  private static void rmr(String path) throws Exception {
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
  }

  public static double round(double val, int places) {
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

  public static Vector tweakValue(Vector point) {
    return point.plus(0.1);

  }

  public static void referenceFuzzyKMeans(List<Vector> points,
      List<SoftCluster> clusterList, Map<String, String> pointClusterInfo,
      String distanceMeasureClass, double threshold, int numIter)
      throws Exception {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(distanceMeasureClass);

    DistanceMeasure measure = (DistanceMeasure) cl.newInstance();
    SoftCluster.config(measure, threshold);
    boolean converged = false;
    for (int iter = 0; !converged && iter < numIter; iter++) {
      converged = iterateReference(points, clusterList, measure);
    }
    computeCluster(points, clusterList, measure, pointClusterInfo);
  }

  public static boolean iterateReference(List<Vector> points,
      List<SoftCluster> clusterList, DistanceMeasure measure) {
    // for each
    for (Vector point : points) {
      List<Double> clusterDistanceList = new ArrayList<Double>();
      for (SoftCluster cluster : clusterList) {
        clusterDistanceList.add(measure.distance(point, cluster.getCenter()));
      }

      for (int i = 0; i < clusterList.size(); i++) {
        double probWeight = SoftCluster.computeProbWeight(clusterDistanceList
            .get(i), clusterDistanceList);
        clusterList.get(i).addPoint(point,
            Math.pow(probWeight, SoftCluster.getM()));
      }
    }
    boolean converged = true;
    for (SoftCluster cluster : clusterList) {
      if (!cluster.computeConvergence())
        converged = false;
    }
    // update the cluster centers
    if (!converged)
      for (SoftCluster cluster : clusterList)
        cluster.recomputeCenter();
    return converged;

  }

  public static void computeCluster(List<Vector> points,
      List<SoftCluster> clusterList, DistanceMeasure measure,
      Map<String, String> pointClusterInfo) {

    for (Vector point : points) {
      StringBuilder outputValue = new StringBuilder("[");
      List<Double> clusterDistanceList = new ArrayList<Double>();
      for (SoftCluster cluster : clusterList) {
        clusterDistanceList.add(measure.distance(point, cluster.getCenter()));
      }
      for (int i = 0; i < clusterList.size(); i++) {
        double probWeight = SoftCluster.computeProbWeight(clusterDistanceList
            .get(i), clusterDistanceList);
        outputValue.append(clusterList.get(i).getClusterId()).append(':')
            .append(probWeight).append(' ');
      }

      pointClusterInfo.put(point.asFormatString().trim(), outputValue
          .toString().trim()
          + ']');
    }
  }

  public void testReferenceImplementation() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);
    for (int k = 0; k < points.size(); k++) {
      System.out.println("test k= " + k);

      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();
      // pick k initial cluster centers at random
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));
        SoftCluster cluster = new SoftCluster(vec);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter(), 1);

        clusterList.add(cluster);
      }
      Map<String, String> pointClusterInfo = new HashMap<String, String>();
      // run reference FuzzyKmeans algorithm
      referenceFuzzyKMeans(points, clusterList, pointClusterInfo,
          EuclideanDistanceMeasure.class.getName(), 0.001, 2);

      // iterate for each point
      for (String value : pointClusterInfo.values()) {
        String clusterInfoStr = value.substring(1, value.length() - 1);
        String[] clusterInfoList = clusterInfoStr.split(" ");
        assertEquals("Number of clusters", k + 1, clusterInfoList.length);
        double prob = 0.0;
        for (String clusterInfo : clusterInfoList) {
          String[] clusterProb = clusterInfo.split(":");

          double clusterProbVal = Double.parseDouble(clusterProb[1]);
          prob += clusterProbVal;
        }
        prob = round(prob, 1);
        assertEquals(
            "Sum of cluster Membership problability should be equal to=", 1.0,
            prob);
      }
    }
  }

  public void testFuzzyKMeansMRJob() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    testData = new File("testdata/points");
    if (!testData.exists())
      testData.mkdir();

    TestKmeansClustering.writePointsToFile(points, "testdata/points/file1");
    TestKmeansClustering.writePointsToFile(points, "testdata/points/file2");

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      JobConf job = new JobConf(FuzzyKMeansDriver.class);
      FileSystem fs = FileSystem.get(job);
      Path path = new Path("testdata/clusters");
      if (fs.exists(path)) {
        fs.delete(path, true);
      }

      testData = new File("testdata/clusters");
      if (!testData.exists())
        testData.mkdir();

      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
          new FileOutputStream("testdata/clusters/part-00000"), Charset
              .forName("UTF-8")));

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));

        SoftCluster cluster = new SoftCluster(vec);
        // add the center so the centroid will be correct upon output
        cluster.addPoint(cluster.getCenter(), 1);
        writer.write(cluster.getIdentifier() + '\t'
            + SoftCluster.formatCluster(cluster) + '\n');

      }
      writer.flush();
      writer.close();

      JobConf conf = new JobConf(FuzzyKMeansDriver.class);
      Path outPath = new Path("output");
      fs = FileSystem.get(conf);
      if (fs.exists(outPath)) {
        fs.delete(outPath, true);
      }
      fs.mkdirs(outPath);
      // now run the Job      
      FuzzyKMeansDriver.runJob("testdata/points", "testdata/clusters",
          "output", EuclideanDistanceMeasure.class.getName(), 0.001, 2, 1,
          k + 1, 2);      

      // now compare the expected clusters with actual
      File outDir = new File("output/points");
      assertTrue("output dir exists?", outDir.exists());
      String[] outFiles = outDir.list();
//      assertEquals("output dir files?", 4, outFiles.length);
      BufferedReader reader = new BufferedReader(new InputStreamReader(
          new FileInputStream("output/points/part-00000"), Charset
              .forName("UTF-8")));

      while (reader.ready()) {
        String line = reader.readLine();
        String[] lineParts = line.split("\t");
        assertEquals("line parts", 2, lineParts.length);
        String clusterInfoStr = lineParts[1].replace("[", "").replace("]", "");

        String[] clusterInfoList = clusterInfoStr.split(" ");
        assertEquals("Number of clusters", k + 1, clusterInfoList.length);
        double prob = 0.0;
        for (String clusterInfo : clusterInfoList) {
          String[] clusterProb = clusterInfo.split(":");

          double clusterProbVal = Double.parseDouble(clusterProb[1]);
          prob += clusterProbVal;
        }
        prob = round(prob, 1);
        assertEquals(
            "Sum of cluster Membership problability should be equal to=", 1.0,
            prob);
      }

      reader.close();

    }

  }

  public void testFuzzyKMeansMapper() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);

    DistanceMeasure measure = new EuclideanDistanceMeasure();
    SoftCluster.config(measure, 0.001);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));

        SoftCluster cluster = new SoftCluster(vec);
        cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
      for (Vector point : points) {
        mapper.map(new Text(), new Text(point.asFormatString()), mapCollector,
            null);
      }

      // now verify mapper output
      assertEquals("Mapper Keys", k + 1, mapCollector.getData().size());

      Map<String, Double> pointTotalProbMap = new HashMap<String, Double>();

      for (String key : mapCollector.getKeys()) {
        // SoftCluster cluster = SoftCluster.decodeCluster(key);
        List<Text> values = mapCollector.getValue(key);

        for (Text value : values) {
          String pointInfo = value.toString();
          double pointProb = Double.parseDouble(pointInfo.substring(0,
              pointInfo.indexOf(FuzzyKMeansDriver.MAPPER_VALUE_SEPARATOR)));

          String encodedVector = pointInfo.substring(pointInfo
              .indexOf(FuzzyKMeansDriver.MAPPER_VALUE_SEPARATOR) + 1);

          Double val = pointTotalProbMap.get(encodedVector);
          double probVal = 0.0;
          if (val != null) {
            probVal = val;
          }

          pointTotalProbMap.put(encodedVector, probVal + pointProb);
        }
      }

      for (Map.Entry<String, Double> entry : pointTotalProbMap.entrySet()) {
        String key = entry.getKey();
        double value = round(entry.getValue(), 1);

        assertEquals("total Prob for Point:" + key, 1.0, value);
      }
    }
  }

  public void testFuzzyKMeansCombiner() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);

    DistanceMeasure measure = new EuclideanDistanceMeasure();
    SoftCluster.config(measure, 0.001);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));

        SoftCluster cluster = new SoftCluster(vec);
        cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
      for (Vector point : points) {
        mapper.map(new Text(), new Text(point.asFormatString()), mapCollector,
            null);
      }

      // run combiner
      DummyOutputCollector<Text, Text> combinerCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();

      for (String key : mapCollector.getKeys()) {

        List<Text> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector,
            null);
      }

      // now verify the combiner output
      assertEquals("Combiner Output", k + 1, combinerCollector.getData().size());

      for (String key : combinerCollector.getKeys()) {
        List<Text> values = combinerCollector.getValue(key);
        assertEquals("too many values", 1, values.size());
      }
    }
  }

  public void testFuzzyKMeansReducer() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);

    DistanceMeasure measure = new EuclideanDistanceMeasure();
    SoftCluster.config(measure, 0.001);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));

        SoftCluster cluster = new SoftCluster(vec, i);
        // cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
      for (Vector point : points) {
        mapper.map(new Text(), new Text(point.asFormatString()), mapCollector,
            null);
      }

      // run combiner
      DummyOutputCollector<Text, Text> combinerCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();

      for (String key : mapCollector.getKeys()) {

        List<Text> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector,
            null);
      }

      // run reducer
      DummyOutputCollector<Text, Text> reducerCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      reducer.config(clusterList);

      for (String key : combinerCollector.getKeys()) {
        List<Text> values = combinerCollector.getValue(key);
        reducer
            .reduce(new Text(key), values.iterator(), reducerCollector, null);
      }

      // now verify the reducer output
      assertEquals("Reducer Output", k + 1, combinerCollector.getData().size());

      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = new ArrayList<SoftCluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));
        reference.add(new SoftCluster(vec, i));
      }
      iterateReference(points, reference, measure);
      for (SoftCluster key : reference) {
        String clusterId = key.getIdentifier();
        List<Text> values = reducerCollector.getValue(clusterId);
        SoftCluster cluster = SoftCluster.decodeCluster(values.get(0)
            .toString());
        System.out.println("ref= " + key.toString() + " cluster= "
            + cluster.toString());
        assertEquals(k + " center[" + key + "][0]", key.getCenter().get(0),
            cluster.getCenter().get(0));
        assertEquals(k + " center[" + key + "][1]", key.getCenter().get(1),
            cluster.getCenter().get(1));
      }
    }
  }

  public void testFuzzyKMeansClusterMapper() throws Exception {
    List<Vector> points = TestKmeansClustering
        .getPoints(TestKmeansClustering.reference);

    DistanceMeasure measure = new EuclideanDistanceMeasure();
    SoftCluster.config(measure, 0.001);

    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      List<SoftCluster> clusterList = new ArrayList<SoftCluster>();

      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));

        SoftCluster cluster = new SoftCluster(vec, i);
        cluster.addPoint(cluster.getCenter(), 1);
        clusterList.add(cluster);
      }

      // run mapper
      FuzzyKMeansMapper mapper = new FuzzyKMeansMapper();
      mapper.config(clusterList);

      DummyOutputCollector<Text, Text> mapCollector = new DummyOutputCollector<Text, Text>();
      for (Vector point : points) {
        mapper.map(new Text(), new Text(point.asFormatString()), mapCollector,
            null);
      }

      // run combiner
      DummyOutputCollector<Text, Text> combinerCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();

      for (String key : mapCollector.getKeys()) {

        List<Text> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector,
            null);
      }

      // run reducer
      DummyOutputCollector<Text, Text> reducerCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansReducer reducer = new FuzzyKMeansReducer();
      reducer.config(clusterList);

      for (String key : combinerCollector.getKeys()) {
        List<Text> values = combinerCollector.getValue(key);
        reducer
            .reduce(new Text(key), values.iterator(), reducerCollector, null);
      }

      // run clusterMapper
      List<SoftCluster> reducerCluster = new ArrayList<SoftCluster>();

      for (String key : reducerCollector.getKeys()) {
        List<Text> values = reducerCollector.getValue(key);
        reducerCluster.add(SoftCluster.decodeCluster(values.get(0).toString()));
      }

      DummyOutputCollector<Text, Text> clusterMapperCollector = new DummyOutputCollector<Text, Text>();
      FuzzyKMeansClusterMapper clusterMapper = new FuzzyKMeansClusterMapper();
      clusterMapper.config(reducerCluster);
      for (Vector point : points) {
        clusterMapper.map(new Text(), new Text(point.asFormatString()),
            clusterMapperCollector, null);
      }

      // now run for one iteration of referencefuzzykmeans and compare the
      // results
      // compute the reference result after one iteration and compare
      List<SoftCluster> reference = new ArrayList<SoftCluster>();
      for (int i = 0; i < k + 1; i++) {
        Vector vec = tweakValue(points.get(i));
        reference.add(new SoftCluster(vec, i));
      }
      Map<String, String> pointClusterInfo = new HashMap<String, String>();
      referenceFuzzyKMeans(points, reference, pointClusterInfo,
          EuclideanDistanceMeasure.class.getName(), 0.001, 1);

      // Now compare the clustermapper results with reducer
      for (String key : clusterMapperCollector.getKeys()) {
        List<Text> value = clusterMapperCollector.getValue(key);

        String refValue = pointClusterInfo.get(key);
        String clusterInfoStr = refValue.substring(1, refValue.length() - 1);
        String[] refClusterInfoList = clusterInfoStr.split(" ");
        assertEquals("Number of clusters", k + 1, refClusterInfoList.length);
        Map<String, Double> refClusterInfoMap = new HashMap<String, Double>();
        for (String clusterInfo : refClusterInfoList) {
          String[] clusterProb = clusterInfo.split(":");
          double clusterProbVal = Double.parseDouble(clusterProb[1]);
          refClusterInfoMap.put(clusterProb[0], clusterProbVal);
        }

        String[] clusterInfoList = value.get(0).toString().replace("[", "")
            .replace("]", "").split(" ");
        assertEquals("Number of clusters", k + 1, clusterInfoList.length);
        for (String clusterInfo : refClusterInfoList) {
          String[] clusterProb = clusterInfo.split(":");
          double clusterProbVal = Double.parseDouble(clusterProb[1]);
          assertEquals(k + " point:" + key + ": Cluster:" + clusterProb[0],
              refClusterInfoMap.get(clusterProb[0]), clusterProbVal);
        }
      }
    }
  }

}
