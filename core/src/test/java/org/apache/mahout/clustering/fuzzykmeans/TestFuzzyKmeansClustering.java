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
import org.apache.mahout.clustering.kmeans.TestKmeansClustering;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.DummyReporter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
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
  
  private static void computeCluster(List<NamedVector> points,
                                     List<SoftCluster> clusterList,
                                     FuzzyKMeansClusterer clusterer,
                                     Map<String,String> pointClusterInfo) {
    
    for (NamedVector point : points) {
      StringBuilder outputValue = new StringBuilder("[");
      List<Double> clusterDistanceList = new ArrayList<Double>();
      for (SoftCluster cluster : clusterList) {
        clusterDistanceList.add(clusterer.getMeasure().distance(point, cluster.getCenter()));
      }
      for (int i = 0; i < clusterList.size(); i++) {
        double probWeight = clusterer.computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
        outputValue.append(clusterList.get(i).getId()).append(':').append(probWeight).append(' ');
      }
      pointClusterInfo.put(point.getName(), outputValue.toString().trim() + ']');
    }
  }
  
  public void testReferenceImplementation() throws Exception {
    List<NamedVector> points = TestKmeansClustering.getPoints(TestKmeansClustering.reference);
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
      Map<String,String> pointClusterInfo = new HashMap<String,String>();
      // run reference FuzzyKmeans algorithm
      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(points, clusterList,
        new EuclideanDistanceMeasure(), 0.001, 2, 2);
      computeCluster(points, clusters.get(clusters.size() - 1), new FuzzyKMeansClusterer(
          new EuclideanDistanceMeasure(), 0.001, 2), pointClusterInfo);
      
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
        assertEquals("Sum of cluster Membership problability should be equal to=", 1.0, prob);
      }
    }
  }
  
  public void testFuzzyKMeansMRJob() throws Exception {
    List<VectorWritable> points = TestKmeansClustering.getPointsWritable(TestKmeansClustering.reference);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    testData = new File("testdata/points");
    if (!testData.exists()) {
      testData.mkdir();
    }
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, "testdata/points/file1", fs, conf);
    
    for (int k = 0; k < points.size(); k++) {
      System.out.println("testKFuzzyKMeansMRJob k= " + k);
      // pick k initial cluster centers at random
      JobConf job = new JobConf(FuzzyKMeansDriver.class);
      Path path = new Path("testdata/clusters");
      FileSystem fs = FileSystem.get(path.toUri(), job);
      if (fs.exists(path)) {
        fs.delete(path, true);
      }
      
      testData = new File("testdata/clusters");
      if (!testData.exists()) {
        testData.mkdir();
      }
      
      /*
       * BufferedWriter writer = new BufferedWriter(new OutputStreamWriter( new
       * FileOutputStream("testdata/clusters/part-00000"), Charset .forName("UTF-8")));
       */
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
          new Path("testdata/clusters/part-00000"), Text.class, SoftCluster.class);
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
      
      Path outPath = new Path("output");
      fs = FileSystem.get(outPath.toUri(), conf);
      if (fs.exists(outPath)) {
        fs.delete(outPath, true);
      }
      fs.mkdirs(outPath);
      // now run the Job
      FuzzyKMeansDriver.runJob("testdata/points", "testdata/clusters", "output",
        EuclideanDistanceMeasure.class.getName(), 0.001, 2, 1, k + 1, 2);
      
      // now compare the expected clusters with actual
      File outDir = new File("output/points");
      assertTrue("output dir exists?", outDir.exists());
      outDir.list();
      // assertEquals("output dir files?", 4, outFiles.length);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path("output/points/part-00000"), conf);
      IntWritable key = new IntWritable();
      VectorWritable out = new VectorWritable();
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
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      mapper.configure(conf);
      
      DummyOutputCollector<Text,FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }
      
      // now verify mapper output
      assertEquals("Mapper Keys", k + 1, mapCollector.getData().size());
      
      Map<Vector,Double> pointTotalProbMap = new HashMap<Vector,Double>();
      
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
      
      for (Map.Entry<Vector,Double> entry : pointTotalProbMap.entrySet()) {
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
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      mapper.configure(conf);
      
      DummyOutputCollector<Text,FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }
      
      // run combiner
      DummyOutputCollector<Text,FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
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
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      mapper.configure(conf);
      
      DummyOutputCollector<Text,FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }
      
      // run combiner
      DummyOutputCollector<Text,FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      combiner.configure(conf);
      
      for (Text key : mapCollector.getKeys()) {
        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector, null);
      }
      
      // run reducer
      DummyOutputCollector<Text,SoftCluster> reducerCollector = new DummyOutputCollector<Text,SoftCluster>();
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
      List<NamedVector> pointsVectors = new ArrayList<NamedVector>();
      for (VectorWritable point : points) {
        pointsVectors.add((NamedVector) point.get());
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
      conf.set(FuzzyKMeansConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
      conf.set(FuzzyKMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.001");
      conf.set(FuzzyKMeansConfigKeys.M_KEY, "2");
      mapper.configure(conf);
      
      DummyOutputCollector<Text,FuzzyKMeansInfo> mapCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      for (VectorWritable point : points) {
        mapper.map(new Text(), point, mapCollector, null);
      }
      for (SoftCluster softCluster : clusterList) {
        softCluster.recomputeCenter();
      }
      // run combiner
      DummyOutputCollector<Text,FuzzyKMeansInfo> combinerCollector = new DummyOutputCollector<Text,FuzzyKMeansInfo>();
      FuzzyKMeansCombiner combiner = new FuzzyKMeansCombiner();
      combiner.configure(conf);
      
      for (Text key : mapCollector.getKeys()) {
        
        List<FuzzyKMeansInfo> values = mapCollector.getValue(key);
        combiner.reduce(new Text(key), values.iterator(), combinerCollector, null);
      }
      
      // run reducer
      DummyOutputCollector<Text,SoftCluster> reducerCollector = new DummyOutputCollector<Text,SoftCluster>();
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
      
      DummyOutputCollector<IntWritable,VectorWritable> clusterMapperCollector = new DummyOutputCollector<IntWritable,VectorWritable>();
      
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
      Map<String,String> pointClusterInfo = new HashMap<String,String>();
      List<NamedVector> pointsVectors = new ArrayList<NamedVector>();
      for (VectorWritable point : points) {
        pointsVectors.add((NamedVector) point.get());
      }
      
      List<List<SoftCluster>> clusters = FuzzyKMeansClusterer.clusterPoints(pointsVectors, reference,
        new EuclideanDistanceMeasure(), 0.001, 2, 1);
      computeCluster(pointsVectors, clusters.get(clusters.size() - 1), new FuzzyKMeansClusterer(
          new EuclideanDistanceMeasure(), 0.001, 2), pointClusterInfo);
      
      // Now compare the clustermapper results with reducer
      for (IntWritable key : clusterMapperCollector.getKeys()) {
        List<VectorWritable> value = clusterMapperCollector.getValue(key);
        
        String refValue = pointClusterInfo.get(key.toString());
        String clusterInfoStr = refValue.substring(1, refValue.length() - 1);
        String[] refClusterInfoList = clusterInfoStr.split(" ");
        assertEquals("Number of clusters", k + 1, refClusterInfoList.length);
        Map<String,Double> refClusterInfoMap = new HashMap<String,Double>();
        for (String clusterInfo : refClusterInfoList) {
          String[] clusterProb = clusterInfo.split(":");
          double clusterProbVal = Double.parseDouble(clusterProb[1]);
          refClusterInfoMap.put(clusterProb[0], clusterProbVal);
        }
        
        VectorWritable kMeansOutput = value.get(0);
        // TODO: fail("test this output");
      }
    }
  }
  
}
