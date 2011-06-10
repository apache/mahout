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

package org.apache.mahout.clustering.meanshift;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.common.kernel.IKernelProfile;
import org.apache.mahout.common.kernel.TriangularKernelProfile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestMeanShift extends MahoutTestCase {
  
  private Vector[] raw = null;
  
  // DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();
  
  private final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();
  private final IKernelProfile kernelProfile = new TriangularKernelProfile();
  
  /**
   * Print the canopies to the transcript
   * 
   * @param canopies
   *          a List<Canopy>
   */
  private static void printCanopies(Iterable<MeanShiftCanopy> canopies) {
    for (MeanShiftCanopy canopy : canopies) {
      System.out.println(canopy.asFormatString(null));
    }
  }
  
  /**
   * Print a graphical representation of the clustered image points as a 10x10
   * character mask
   */
  private void printImage(Iterable<MeanShiftCanopy> canopies) {
    char[][] out = new char[10][10];
    for (int i = 0; i < out.length; i++) {
      for (int j = 0; j < out[0].length; j++) {
        out[i][j] = ' ';
      }
    }
    for (MeanShiftCanopy canopy : canopies) {
      int ch = 'A' + canopy.getId();
      for (int pid : canopy.getBoundPoints().toList()) {
        Vector pt = raw[pid];
        out[(int) pt.getQuick(0)][(int) pt.getQuick(1)] = (char) ch;
      }
    }
    for (char[] anOut : out) {
      System.out.println(anOut);
    }
  }
  
  private List<MeanShiftCanopy> getInitialCanopies() {
    int nextCanopyId = 0;
    List<MeanShiftCanopy> canopies = Lists.newArrayList();
    for (Vector point : raw) {
      canopies.add(new MeanShiftCanopy(point, nextCanopyId++,
          euclideanDistanceMeasure));
    }
    return canopies;
  }
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    raw = new Vector[100];
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        int ix = i * 10 + j;
        Vector v = new DenseVector(3);
        v.setQuick(0, i);
        v.setQuick(1, j);
        if (i == j) {
          v.setQuick(2, 9);
        } else if (i + j == 9) {
          v.setQuick(2, 4.5);
        }
        raw[ix] = v;
      }
    }
  }
  
  /**
   * Story: User can exercise the reference implementation to verify that the
   * test datapoints are clustered in a reasonable manner.
   */
  @Test
  public void testReferenceImplementation() {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(
        new EuclideanDistanceMeasure(), new TriangularKernelProfile(), 4.0,
        1.0, 0.5);
    List<MeanShiftCanopy> canopies = Lists.newArrayList();
    // add all points to the canopies
    int nextCanopyId = 0;
    for (Vector aRaw : raw) {
      clusterer.mergeCanopy(new MeanShiftCanopy(aRaw, nextCanopyId++,
          euclideanDistanceMeasure), canopies);
    }
    boolean done = false;
    int iter = 1;
    while (!done) {// shift canopies to their centroids
      done = true;
      List<MeanShiftCanopy> migratedCanopies = Lists.newArrayList();
      for (MeanShiftCanopy canopy : canopies) {
        done = clusterer.shiftToMean(canopy) && done;
        clusterer.mergeCanopy(canopy, migratedCanopies);
      }
      canopies = migratedCanopies;
      printCanopies(canopies);
      printImage(canopies);
      System.out.println(iter++);
    }
  }
  
  /**
   * Test the MeanShiftCanopyClusterer's reference implementation. Should
   * produce the same final output as above.
   */
  @Test
  public void testClustererReferenceImplementation() {
    Iterable<Vector> points = Lists.newArrayList(raw);
    List<MeanShiftCanopy> canopies = MeanShiftCanopyClusterer.clusterPoints(
        points, euclideanDistanceMeasure, kernelProfile, 0.5, 4, 1, 10);
    printCanopies(canopies);
    printImage(canopies);
  }
  
  /**
   * Story: User can produce initial canopy centers using a
   * EuclideanDistanceMeasure and a CanopyMapper/Combiner which clusters input
   * points to produce an output set of canopies.
   */
  @Test
  public void testCanopyMapperEuclidean() throws Exception {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(
        euclideanDistanceMeasure, kernelProfile, 4, 1, 0.5);
    // get the initial canopies
    List<MeanShiftCanopy> canopies = getInitialCanopies();
    // build the reference set
    Collection<MeanShiftCanopy> refCanopies = Lists.newArrayList();
    int nextCanopyId = 0;
    for (Vector aRaw : raw) {
      clusterer.mergeCanopy(new MeanShiftCanopy(aRaw, nextCanopyId++,
          euclideanDistanceMeasure), refCanopies);
    }
    
    Configuration conf = new Configuration();
    conf.set(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(MeanShiftCanopyConfigKeys.KERNEL_PROFILE_KEY,
        "org.apache.mahout.common.kernel.TriangularKernelProfile");
    conf.set(MeanShiftCanopyConfigKeys.T1_KEY, "4");
    conf.set(MeanShiftCanopyConfigKeys.T2_KEY, "1");
    conf.set(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.5");
    
    // map the data
    MeanShiftCanopyMapper mapper = new MeanShiftCanopyMapper();
    DummyRecordWriter<Text,MeanShiftCanopy> mapWriter = new DummyRecordWriter<Text,MeanShiftCanopy>();
    Mapper<WritableComparable<?>,MeanShiftCanopy,Text,MeanShiftCanopy>.Context mapContext = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    mapper.setup(mapContext);
    for (MeanShiftCanopy canopy : canopies) {
      mapper.map(new Text(), canopy, mapContext);
    }
    mapper.cleanup(mapContext);
    
    // now verify the output
    assertEquals("Number of map results", 1, mapWriter.getData().size());
    List<MeanShiftCanopy> data = mapWriter.getValue(new Text("canopy"));
    assertEquals("Number of canopies", refCanopies.size(), data.size());
    
    // add all points to the reference canopies
    Map<String,MeanShiftCanopy> refCanopyMap = Maps.newHashMap();
    for (MeanShiftCanopy canopy : refCanopies) {
      clusterer.shiftToMean(canopy);
      refCanopyMap.put(canopy.getIdentifier(), canopy);
    }
    // build a map of the combiner output
    Map<String,MeanShiftCanopy> canopyMap = Maps.newHashMap();
    for (MeanShiftCanopy d : data) {
      canopyMap.put(d.getIdentifier(), d);
    }
    // compare the maps
    for (Map.Entry<String,MeanShiftCanopy> stringMeanShiftCanopyEntry : refCanopyMap
        .entrySet()) {
      MeanShiftCanopy ref = stringMeanShiftCanopyEntry.getValue();
      
      MeanShiftCanopy canopy = canopyMap.get((ref.isConverged() ? "MSV-"
          : "MSC-") + ref.getId());
      assertEquals("ids", ref.getId(), canopy.getId());
      assertEquals("centers(" + ref.getIdentifier() + ')', ref.getCenter()
          .asFormatString(), canopy.getCenter().asFormatString());
      assertEquals("bound points", ref.getBoundPoints().toList().size(), canopy
          .getBoundPoints().toList().size());
    }
  }
  
  /**
   * Story: User can produce final canopy centers using a
   * EuclideanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   */
  @Test
  public void testCanopyReducerEuclidean() throws Exception {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(
        euclideanDistanceMeasure, kernelProfile, 4, 1, 0.5);
    // get the initial canopies
    List<MeanShiftCanopy> canopies = getInitialCanopies();
    // build the mapper output reference set
    Collection<MeanShiftCanopy> mapperReference = Lists.newArrayList();
    int nextCanopyId = 0;
    for (Vector aRaw : raw) {
      clusterer.mergeCanopy(new MeanShiftCanopy(aRaw, nextCanopyId++,
          euclideanDistanceMeasure), mapperReference);
    }
    for (MeanShiftCanopy canopy : mapperReference) {
      clusterer.shiftToMean(canopy);
    }
    // build the reducer reference output set
    Collection<MeanShiftCanopy> reducerReference = Lists.newArrayList();
    for (MeanShiftCanopy canopy : mapperReference) {
      clusterer.mergeCanopy(canopy, reducerReference);
    }
    for (MeanShiftCanopy canopy : reducerReference) {
      clusterer.shiftToMean(canopy);
    }
    
    Configuration conf = new Configuration();
    conf.set(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(MeanShiftCanopyConfigKeys.KERNEL_PROFILE_KEY,
        "org.apache.mahout.common.kernel.TriangularKernelProfile");
    conf.set(MeanShiftCanopyConfigKeys.T1_KEY, "4");
    conf.set(MeanShiftCanopyConfigKeys.T2_KEY, "1");
    conf.set(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY, "0.5");
    conf.set(MeanShiftCanopyConfigKeys.CONTROL_PATH_KEY, "output/control");
    
    MeanShiftCanopyMapper mapper = new MeanShiftCanopyMapper();
    DummyRecordWriter<Text,MeanShiftCanopy> mapWriter = new DummyRecordWriter<Text,MeanShiftCanopy>();
    Mapper<WritableComparable<?>,MeanShiftCanopy,Text,MeanShiftCanopy>.Context mapContext = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    mapper.setup(mapContext);
    
    // map the data
    for (MeanShiftCanopy canopy : canopies) {
      mapper.map(new Text(), canopy, mapContext);
    }
    mapper.cleanup(mapContext);
    
    assertEquals("Number of map results", 1, mapWriter.getData().size());
    // now reduce the mapper output
    MeanShiftCanopyReducer reducer = new MeanShiftCanopyReducer();
    DummyRecordWriter<Text,MeanShiftCanopy> reduceWriter = new DummyRecordWriter<Text,MeanShiftCanopy>();
    Reducer<Text,MeanShiftCanopy,Text,MeanShiftCanopy>.Context reduceContext = DummyRecordWriter
        .build(reducer, conf, reduceWriter, Text.class, MeanShiftCanopy.class);
    reducer.setup(reduceContext);
    reducer.reduce(new Text("canopy"), mapWriter.getValue(new Text("canopy")),
        reduceContext);
    reducer.cleanup(reduceContext);
    
    // now verify the output
    assertEquals("Number of canopies", reducerReference.size(), reduceWriter
        .getKeys().size());
    
    // add all points to the reference canopy maps
    Map<String,MeanShiftCanopy> reducerReferenceMap = Maps.newHashMap();
    for (MeanShiftCanopy canopy : reducerReference) {
      reducerReferenceMap.put(canopy.getIdentifier(), canopy);
    }
    // compare the maps
    for (Map.Entry<String,MeanShiftCanopy> mapEntry : reducerReferenceMap
        .entrySet()) {
      MeanShiftCanopy refCanopy = mapEntry.getValue();
      
      List<MeanShiftCanopy> values = reduceWriter.getValue(new Text((refCanopy
          .isConverged() ? "MSV-" : "MSC-") + refCanopy.getId()));
      assertEquals("values", 1, values.size());
      MeanShiftCanopy reducerCanopy = values.get(0);
      assertEquals("ids", refCanopy.getId(), reducerCanopy.getId());
      long refNumPoints = refCanopy.getNumPoints();
      long reducerNumPoints = reducerCanopy.getNumPoints();
      assertEquals("numPoints", refNumPoints, reducerNumPoints);
      String refCenter = refCanopy.getCenter().asFormatString();
      String reducerCenter = reducerCanopy.getCenter().asFormatString();
      assertEquals("centers(" + mapEntry.getKey() + ')', refCenter,
          reducerCenter);
      assertEquals("bound points", refCanopy.getBoundPoints().toList().size(),
          reducerCanopy.getBoundPoints().toList().size());
    }
  }
  
  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure.
   */
  @Test
  public void testCanopyEuclideanMRJob() throws Exception {
    Path input = getTestTempDirPath("testdata");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    Collection<VectorWritable> points = Lists.newArrayList();
    for (Vector v : raw) {
      points.add(new VectorWritable(v));
    }
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job using the run() command. Other tests can continue to use
    // runJob().
    Path output = getTestTempDirPath("output");
    // MeanShiftCanopyDriver.runJob(input, output,
    // EuclideanDistanceMeasure.class.getName(), 4, 1, 0.5, 10, false, false);
    String[] args = {optKey(DefaultOptionCreator.INPUT_OPTION),
        getTestTempDirPath("testdata").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
        EuclideanDistanceMeasure.class.getName(),
        optKey(DefaultOptionCreator.KERNEL_PROFILE_OPTION),
        TriangularKernelProfile.class.getName(),
        optKey(DefaultOptionCreator.T1_OPTION), "4",
        optKey(DefaultOptionCreator.T2_OPTION), "1",
        optKey(DefaultOptionCreator.CLUSTERING_OPTION),
        optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "7",
        optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.2",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION)};
    ToolRunner.run(conf, new MeanShiftCanopyDriver(), args);
    Path outPart = new Path(output, "clusters-4/part-r-00000");
    long count = HadoopUtil.countRecords(outPart, conf);
    assertEquals("count", 3, count);
    outPart = new Path(output, "clusters-0/part-m-00000");
    Iterator<?> iterator = new SequenceFileValueIterator<Writable>(outPart,
        true, conf);
    // now test the initial clusters to ensure the type of their centers has
    // been retained
    while (iterator.hasNext()) {
      Cluster canopy = (Cluster) iterator.next();
      assertTrue(canopy.getCenter() instanceof DenseVector);
    }
  }
  
  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure.
   */
  @Test
  public void testCanopyEuclideanSeqJob() throws Exception {
    Path input = getTestTempDirPath("testdata");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    Collection<VectorWritable> points = Lists.newArrayList();
    for (Vector v : raw) {
      points.add(new VectorWritable(v));
    }
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job using the run() command. Other tests can continue to use
    // runJob().
    Path output = getTestTempDirPath("output");
    System.out.println("Output Path: " + output);
    // MeanShiftCanopyDriver.runJob(input, output,
    // EuclideanDistanceMeasure.class.getName(), 4, 1, 0.5, 10, false, false);
    String[] args = {optKey(DefaultOptionCreator.INPUT_OPTION),
        getTestTempDirPath("testdata").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
        EuclideanDistanceMeasure.class.getName(),
        optKey(DefaultOptionCreator.KERNEL_PROFILE_OPTION),
        TriangularKernelProfile.class.getName(),
        optKey(DefaultOptionCreator.T1_OPTION), "4",
        optKey(DefaultOptionCreator.T2_OPTION), "1",
        optKey(DefaultOptionCreator.CLUSTERING_OPTION),
        optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "7",
        optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.2",
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.METHOD_OPTION),
        DefaultOptionCreator.SEQUENTIAL_METHOD};
    ToolRunner.run(new Configuration(), new MeanShiftCanopyDriver(), args);
    Path outPart = new Path(output, "clusters-7/part-r-00000");
    long count = HadoopUtil.countRecords(outPart, conf);
    assertEquals("count", 3, count);
  }
}
