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

package org.apache.mahout.clustering.canopy;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.UserDefinedDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestCanopyCreation extends MahoutTestCase {

  private static final double[][] RAW = { { 1, 1 }, { 2, 1 }, { 1, 2 },
      { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 4 }, { 4, 5 }, { 5, 5 } };

  private List<Canopy> referenceManhattan;

  private final DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  private List<Vector> manhattanCentroids;

  private List<Canopy> referenceEuclidean;

  private final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  private List<Vector> euclideanCentroids;

  private FileSystem fs;

  private static List<VectorWritable> getPointsWritable() {
    List<VectorWritable> points = new ArrayList<VectorWritable>();
    for (double[] fr : RAW) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  private static List<Vector> getPoints() {
    List<Vector> points = new ArrayList<Vector>();
    for (double[] fr : RAW) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(vec);
    }
    return points;
  }

  /**
   * Print the canopies to the transcript
   * 
   * @param canopies
   *          a List<Canopy>
   */
  private static void printCanopies(Iterable<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      System.out.println(canopy.asFormatString(null));
    }
  }

  private static Canopy findCanopy(Integer key, Iterable<Canopy> canopies) {
    for (Canopy c : canopies) {
      if (c.getId() == key) {
        return c;
      }
    }
    return null;
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    fs = FileSystem.get(new Configuration());
    referenceManhattan = CanopyClusterer.createCanopies(getPoints(),
        manhattanDistanceMeasure, 3.1, 2.1);
    manhattanCentroids = CanopyClusterer.getCenters(referenceManhattan);
    referenceEuclidean = CanopyClusterer.createCanopies(getPoints(),
        euclideanDistanceMeasure, 3.1, 2.1);
    euclideanCentroids = CanopyClusterer.getCenters(referenceEuclidean);
  }

  /**
   * Story: User can cluster points using a ManhattanDistanceMeasure and a
   * reference implementation
   */
  @Test
  public void testReferenceManhattan() throws Exception {
    // see setUp for cluster creation
    printCanopies(referenceManhattan);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceManhattan.get(canopyIx);
      int[] expectedNumPoints = { 4, 4, 3 };
      double[][] expectedCentroids = { { 1.5, 1.5 }, { 4.0, 4.0 },
          { 4.666666666666667, 4.6666666666666667 } };
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx), EPSILON);
      }
    }
  }

  /**
   * Story: User can cluster points using a EuclideanDistanceMeasure and a
   * reference implementation
   */
  @Test
  public void testReferenceEuclidean() throws Exception {
    // see setUp for cluster creation
    printCanopies(referenceEuclidean);
    assertEquals("number of canopies", 3, referenceEuclidean.size());
    int[] expectedNumPoints = { 5, 5, 3 };
    double[][] expectedCentroids = { { 1.8, 1.8 }, { 4.2, 4.2 },
        { 4.666666666666667, 4.666666666666667 } };
    for (int canopyIx = 0; canopyIx < referenceEuclidean.size(); canopyIx++) {
      Canopy testCanopy = referenceEuclidean.get(canopyIx);
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx), EPSILON);
      }
    }
  }

  /**
   * Story: User can produce initial canopy centers using a
   * ManhattanDistanceMeasure and a CanopyMapper which clusters input points to
   * produce an output set of canopy centroid points.
   */
  @Test
  public void testCanopyMapperManhattan() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, manhattanDistanceMeasure
        .getClass().getName());
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, writer);
    mapper.setup(context);

    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    mapper.cleanup(context);
    assertEquals("Number of map results", 1, writer.getData().size());
    // now verify the output
    List<VectorWritable> data = writer.getValue(new Text("centroid"));
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error",
          manhattanCentroids.get(i).asFormatString(), data.get(i).get()
              .asFormatString());
    }
  }

  /**
   * Story: User can produce initial canopy centers using a
   * EuclideanDistanceMeasure and a CanopyMapper/Combiner which clusters input
   * points to produce an output set of canopy centroid points.
   */
  @Test
  public void testCanopyMapperEuclidean() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, euclideanDistanceMeasure
        .getClass().getName());
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<Text, VectorWritable> writer = new DummyRecordWriter<Text, VectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, writer);
    mapper.setup(context);

    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    mapper.cleanup(context);
    assertEquals("Number of map results", 1, writer.getData().size());
    // now verify the output
    List<VectorWritable> data = writer.getValue(new Text("centroid"));
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error",
          euclideanCentroids.get(i).asFormatString(), data.get(i).get()
              .asFormatString());
    }
  }

  /**
   * Story: User can produce final canopy centers using a
   * ManhattanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   */
  @Test
  public void testCanopyReducerManhattan() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<Text, Canopy> writer = new DummyRecordWriter<Text, Canopy>();
    Reducer<Text, VectorWritable, Text, Canopy>.Context context = DummyRecordWriter
        .build(reducer, conf, writer, Text.class, VectorWritable.class);
    reducer.setup(context);

    List<VectorWritable> points = getPointsWritable();
    reducer.reduce(new Text("centroid"), points, context);
    Set<Text> keys = writer.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (Text key : keys) {
      List<Canopy> data = writer.getValue(key);
      assertEquals(manhattanCentroids.get(i).asFormatString()
          + " is not equal to "
          + data.get(0).computeCentroid().asFormatString(), manhattanCentroids
          .get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a
   * EuclideanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   */
  @Test
  public void testCanopyReducerEuclidean() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<Text, Canopy> writer = new DummyRecordWriter<Text, Canopy>();
    Reducer<Text, VectorWritable, Text, Canopy>.Context context = DummyRecordWriter
        .build(reducer, conf, writer, Text.class, VectorWritable.class);
    reducer.setup(context);

    List<VectorWritable> points = getPointsWritable();
    reducer.reduce(new Text("centroid"), points, context);
    Set<Text> keys = writer.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (Text key : keys) {
      List<Canopy> data = writer.getValue(key);
      assertEquals(euclideanCentroids.get(i).asFormatString()
          + " is not equal to "
          + data.get(0).computeCentroid().asFormatString(), euclideanCentroids
          .get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job
   * and a ManhattanDistanceMeasure.
   */
  @Test
  public void testCanopyGenManhattanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration config = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, config);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, config);
    // now run the Canopy Driver
    Path output = getTestTempDirPath("output");
    CanopyDriver.run(config, getTestTempDirPath("testdata"), output,
        manhattanDistanceMeasure, 3.1, 2.1, false, false);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), config);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, config);
    Writable key = new Text();
    Canopy canopy = new Canopy();
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("1st key", "C-0", key.toString());
    assertEquals("1st x value", 1.5, canopy.getCenter().get(0), EPSILON);
    assertEquals("1st y value", 1.5, canopy.getCenter().get(1), EPSILON);
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("2nd key", "C-1", key.toString());
    assertEquals("2nd x value", 4.333333333333334, canopy.getCenter().get(0),
        EPSILON);
    assertEquals("2nd y value", 4.333333333333334, canopy.getCenter().get(1),
        EPSILON);
    assertFalse("more to come", reader.next(key, canopy));
    reader.close();
  }

  /**
   * Story: User can produce final canopy centers using a Hadoop map/reduce job
   * and a EuclideanDistanceMeasure.
   */
  @Test
  public void testCanopyGenEuclideanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration config = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, config);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, config);
    // now run the Canopy Driver
    Path output = getTestTempDirPath("output");
    CanopyDriver.run(config, getTestTempDirPath("testdata"), output,
        euclideanDistanceMeasure, 3.1, 2.1, false, false);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), config);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, config);
    Writable key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C-0", key.toString());
    assertEquals("1st x value", 1.8, value.getCenter().get(0), EPSILON);
    assertEquals("1st y value", 1.8, value.getCenter().get(1), EPSILON);
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C-1", key.toString());
    assertEquals("2nd x value", 4.433333333333334, value.getCenter().get(0),
        EPSILON);
    assertEquals("2nd y value", 4.433333333333334, value.getCenter().get(1),
        EPSILON);
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterMapper and a
   * ManhattanDistanceMeasure.
   */
  @Test
  public void testClusterMapperManhattan() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<IntWritable, WeightedVectorWritable> writer =
        new DummyRecordWriter<IntWritable, WeightedVectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable>.Context context =
        DummyRecordWriter.build(mapper, conf, writer);
    mapper.setup(context);

    Collection<Canopy> canopies = new ArrayList<Canopy>();
    int nextCanopyId = 0;
    for (Vector centroid : manhattanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++,
          manhattanDistanceMeasure));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    Map<IntWritable, List<WeightedVectorWritable>> data = writer.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Entry<IntWritable, List<WeightedVectorWritable>> stringListEntry : data
        .entrySet()) {
      IntWritable key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key.get(), canopies);
      List<WeightedVectorWritable> pts = stringListEntry.getValue();
      for (WeightedVectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef
            .getVector()));
      }
    }
  }

  /**
   * Story: User can cluster a subset of the points using a ClusterMapper and a
   * EuclideanDistanceMeasure.
   */
  @Test
  public void testClusterMapperEuclidean() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    DummyRecordWriter<IntWritable, WeightedVectorWritable> writer = new DummyRecordWriter<IntWritable, WeightedVectorWritable>();
    Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, writer);
    mapper.setup(context);

    Collection<Canopy> canopies = new ArrayList<Canopy>();
    int nextCanopyId = 0;
    for (Vector centroid : euclideanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++,
          euclideanDistanceMeasure));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPointsWritable();
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, context);
    }
    Map<IntWritable, List<WeightedVectorWritable>> data = writer.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Entry<IntWritable, List<WeightedVectorWritable>> stringListEntry : data
        .entrySet()) {
      IntWritable key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key.get(), canopies);
      List<WeightedVectorWritable> pts = stringListEntry.getValue();
      for (WeightedVectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef
            .getVector()));
      }
    }
  }

  /** Story: User can cluster points using sequential execution */
  @Test
  public void testClusteringManhattanSeq() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration config = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, config);
    // now run the Canopy Driver in sequential mode
    Path output = getTestTempDirPath("output");
    CanopyDriver.run(config, getTestTempDirPath("testdata"), output,
        manhattanDistanceMeasure, 3.1, 2.1, true, true);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");
    int ix = 0;
    for (Canopy value : new SequenceFileValueIterable<Canopy>(path, true,
        config)) {
      assertEquals("Center [" + ix + ']', manhattanCentroids.get(ix), value
          .getCenter());
      ix++;
    }

    path = new Path(output, "clusteredPoints/part-m-0");
    long count = HadoopUtil.countRecords(path, config);
    assertEquals("number of points", points.size(), count);
  }

  /** Story: User can cluster points using sequential execution */
  @Test
  public void testClusteringEuclideanSeq() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration config = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, config);
    // now run the Canopy Driver in sequential mode
    Path output = getTestTempDirPath("output");
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION),
        getTestTempDirPath("testdata").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
        EuclideanDistanceMeasure.class.getName(),
        optKey(DefaultOptionCreator.T1_OPTION), "3.1",
        optKey(DefaultOptionCreator.T2_OPTION), "2.1",
        optKey(DefaultOptionCreator.CLUSTERING_OPTION),
        optKey(DefaultOptionCreator.OVERWRITE_OPTION),
        optKey(DefaultOptionCreator.METHOD_OPTION),
        DefaultOptionCreator.SEQUENTIAL_METHOD };
    new CanopyDriver().run(args);

    // verify output from sequence file
    Path path = new Path(output, "clusters-0/part-r-00000");

    int ix = 0;
    for (Canopy value : new SequenceFileValueIterable<Canopy>(path, true,
        config)) {
      assertEquals("Center [" + ix + ']', euclideanCentroids.get(ix), value
          .getCenter());
      ix++;
    }

    path = new Path(output, "clusteredPoints/part-m-0");
    long count = HadoopUtil.countRecords(path, config);
    assertEquals("number of points", points.size(), count);
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a ManhattanDistanceMeasure.
   */
  @Test
  public void testClusteringManhattanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job
    Path output = getTestTempDirPath("output");
    CanopyDriver.run(conf, getTestTempDirPath("testdata"), output,
        manhattanDistanceMeasure, 3.1, 2.1, true, false);
    Path path = new Path(output, "clusteredPoints/part-m-00000");
    long count = HadoopUtil.countRecords(path, conf);
    assertEquals("number of points", points.size(), count);
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure.
   */
  @Test
  public void testClusteringEuclideanMR() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Job using the run() command. Others can use runJob().
    Path output = getTestTempDirPath("output");
    String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION),
        getTestTempDirPath("testdata").toString(),
        optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
        optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
        EuclideanDistanceMeasure.class.getName(),
        optKey(DefaultOptionCreator.T1_OPTION), "3.1",
        optKey(DefaultOptionCreator.T2_OPTION), "2.1",
        optKey(DefaultOptionCreator.CLUSTERING_OPTION),
        optKey(DefaultOptionCreator.OVERWRITE_OPTION) };
    ToolRunner.run(new Configuration(), new CanopyDriver(), args);
    Path path = new Path(output, "clusteredPoints/part-m-00000");
    long count = HadoopUtil.countRecords(path, conf);
    assertEquals("number of points", points.size(), count);
  }

  /**
   * Story: Clustering algorithm must support arbitrary user defined distance
   * measure
   */
  @Test
  public void testUserDefinedDistanceMeasure() throws Exception {
    List<VectorWritable> points = getPointsWritable();
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file1"), fs, conf);
    ClusteringTestUtils.writePointsToFile(points,
        getTestTempFilePath("testdata/file2"), fs, conf);
    // now run the Canopy Driver. User defined measure happens to be a Manhattan
    // subclass so results are same.
    Path output = getTestTempDirPath("output");
    CanopyDriver.run(conf, getTestTempDirPath("testdata"), output,
        new UserDefinedDistanceMeasure(), 3.1, 2.1, false, false);

    // verify output from sequence file
    Configuration job = new Configuration();
    Path path = new Path(output, "clusters-0/part-r-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Writable key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C-0", key.toString());

    assertEquals("1st x value", 1.5, value.getCenter().get(0), EPSILON);
    assertEquals("1st y value", 1.5, value.getCenter().get(1), EPSILON);
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C-1", key.toString());

    assertEquals("1st x value", 4.333333333333334, value.getCenter().get(0),
        EPSILON);
    assertEquals("1st y value", 4.333333333333334, value.getCenter().get(1),
        EPSILON);
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /**
   * Story: User can set T3 and T4 values to be used by the reducer for its T1
   * and T2 thresholds
   */
  @Test
  public void testCanopyReducerT3T4Configuration() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    Configuration conf = new Configuration();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY,
        "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    conf.set(CanopyConfigKeys.T3_KEY, String.valueOf(1.1));
    conf.set(CanopyConfigKeys.T4_KEY, String.valueOf(0.1));
    DummyRecordWriter<Text, Canopy> writer = new DummyRecordWriter<Text, Canopy>();
    Reducer<Text, VectorWritable, Text, Canopy>.Context context = DummyRecordWriter
        .build(reducer, conf, writer, Text.class, VectorWritable.class);
    reducer.setup(context);
    assertEquals(1.1, reducer.canopyClusterer.getT1(), EPSILON);
    assertEquals(0.1, reducer.canopyClusterer.getT2(), EPSILON);
  }
}
