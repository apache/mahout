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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.DummyReporter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.UserDefinedDistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TestCanopyCreation extends MahoutTestCase {

  private static final double[][] raw = {{1, 1}, {2, 1}, {1, 2}, {2, 2},
      {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};

  private List<Canopy> referenceManhattan;

  private final DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  private List<Vector> manhattanCentroids;

  private List<Canopy> referenceEuclidean;

  private final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  private List<Vector> euclideanCentroids;

  private FileSystem fs;

  private static List<VectorWritable> getPoints(double[][] raw) {
    List<VectorWritable> points = new ArrayList<VectorWritable>();
    int i = 0;
    for (double[] fr : raw) {
      Vector vec = new RandomAccessSparseVector(String.valueOf(i++), fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  /** Verify that the given canopies are equivalent to the referenceManhattan */
  private void verifyManhattanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceManhattan);
  }

  /** Verify that the given canopies are equivalent to the referenceEuclidean */
  private void verifyEuclideanCanopies(List<Canopy> canopies) {
    verifyCanopies(canopies, referenceEuclidean);
  }

  /**
   * Verify that the given canopies are equivalent to the reference. This means the number of canopies is the same, the
   * number of points in each is the same and the centroids are the same.
   */
  private static void verifyCanopies(List<Canopy> canopies,
                                     List<Canopy> reference) {
    assertEquals("number of canopies", reference.size(), canopies.size());
    for (int canopyIx = 0; canopyIx < canopies.size(); canopyIx++) {
      Canopy refCanopy = reference.get(canopyIx);
      Canopy testCanopy = canopies.get(canopyIx);
      assertEquals("canopy points " + canopyIx, refCanopy.getNumPoints(),
          testCanopy.getNumPoints());
      Vector refCentroid = refCanopy.computeCentroid();
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.size(); pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid.get(pointIx), testCentroid.get(pointIx));
      }
    }
  }

  /**
   * Print the canopies to the transcript
   *
   * @param canopies a List<Canopy>
   */
  private static void printCanopies(List<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      System.out.println(canopy.toString());
    }
  }


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
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
    rmr("output");
    rmr("testdata");
    referenceManhattan = populateCanopies(manhattanDistanceMeasure,
        getPoints(raw), 3.1, 2.1);
    manhattanCentroids = populateCentroids(referenceManhattan);
    referenceEuclidean = populateCanopies(euclideanDistanceMeasure,
        getPoints(raw), 3.1, 2.1);
    euclideanCentroids = populateCentroids(referenceEuclidean);
  }

  /**
   * Iterate through the canopies, adding their centroids to a list
   *
   * @param canopies a List<Canopy>
   * @return the List<Vector>
   */
  private static List<Vector> populateCentroids(List<Canopy> canopies) {
    List<Vector> result = new ArrayList<Vector>();
    for (Canopy canopy : canopies) {
      result.add(canopy.computeCentroid());
    }
    return result;
  }

  /**
   * Iterate through the points, adding new canopies. Return the canopies.
   *
   * @param measure a DistanceMeasure to use
   * @param points  a list<Vector> defining the points to be clustered
   * @param t1      the T1 distance threshold
   * @param t2      the T2 distance threshold
   * @return the List<Canopy> created
   */
  private static List<Canopy> populateCanopies(DistanceMeasure measure,
                                       List<VectorWritable> points, double t1, double t2) {
    List<Canopy> canopies = new ArrayList<Canopy>();
    /**
     * Reference Implementation: Given a distance metric, one can create
     * canopies as follows: Start with a list of the data points in any order,
     * and with two distance thresholds, T1 and T2, where T1 > T2. (These
     * thresholds can be set by the user, or selected by cross-validation.) Pick
     * a point on the list and measure its distance to all other points. Put all
     * points that are within distance threshold T1 into a canopy. Remove from
     * the list all points that are within distance threshold T2. Repeat until
     * the list is empty.
     */
    int nextCanopyId = 0;
    while (!points.isEmpty()) {
      Iterator<VectorWritable> ptIter = points.iterator();
      Vector p1 = ptIter.next().get();
      ptIter.remove();
      Canopy canopy = new VisibleCanopy(p1, nextCanopyId++);
      canopies.add(canopy);
      while (ptIter.hasNext()) {
        Vector p2 = ptIter.next().get();
        double dist = measure.distance(p1, p2);
        // Put all points that are within distance threshold T1 into the canopy
        if (dist < t1) {
          canopy.addPoint(p2);
        }
        // Remove from the list all points that are within distance threshold T2
        if (dist < t2) {
          ptIter.remove();
        }
      }
    }
    return canopies;
  }

  /** Story: User can cluster points using a ManhattanDistanceMeasure and a reference implementation */
  public void testReferenceManhattan() throws Exception {
    System.out.println("testReferenceManhattan");
    // see setUp for cluster creation
    printCanopies(referenceManhattan);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceManhattan.get(canopyIx);
      int[] expectedNumPoints = {4, 4, 3};
      double[][] expectedCentroids = {{1.5, 1.5}, {4.0, 4.0},
          {4.666666666666667, 4.6666666666666667}};
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /** Story: User can cluster points using a EuclideanDistanceMeasure and a reference implementation */
  public void testReferenceEuclidean() throws Exception {
    System.out.println("testReferenceEuclidean()");
    // see setUp for cluster creation
    printCanopies(referenceEuclidean);
    assertEquals("number of canopies", 3, referenceManhattan.size());
    for (int canopyIx = 0; canopyIx < referenceManhattan.size(); canopyIx++) {
      Canopy testCanopy = referenceEuclidean.get(canopyIx);
      int[] expectedNumPoints = {5, 5, 3};
      double[][] expectedCentroids = {{1.8, 1.8}, {4.2, 4.2},
          {4.666666666666667, 4.666666666666667}};
      assertEquals("canopy points " + canopyIx, expectedNumPoints[canopyIx],
          testCanopy.getNumPoints());
      double[] refCentroid = expectedCentroids[canopyIx];
      Vector testCentroid = testCanopy.computeCentroid();
      for (int pointIx = 0; pointIx < refCentroid.length; pointIx++) {
        assertEquals("canopy centroid " + canopyIx + '[' + pointIx + ']',
            refCentroid[pointIx], testCentroid.get(pointIx));
      }
    }
  }

  /** Story: User can cluster points without instantiating them all in memory at once */
  public void testIterativeManhattan() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    CanopyClusterer clusterer = new CanopyClusterer(new ManhattanDistanceMeasure(), 3.1, 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (VectorWritable point : points) {
      clusterer.addPointToCanopies(point.get(), canopies, new DummyReporter());
    }

    System.out.println("testIterativeManhattan");
    printCanopies(canopies);
    verifyManhattanCanopies(canopies);
  }

  /** Story: User can cluster points without instantiating them all in memory at once */
  public void testIterativeEuclidean() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    CanopyClusterer clusterer = new CanopyClusterer(new EuclideanDistanceMeasure(), 3.1, 2.1);

    List<Canopy> canopies = new ArrayList<Canopy>();
    for (VectorWritable point : points) {
      clusterer.addPointToCanopies(point.get(), canopies, new DummyReporter());
    }

    System.out.println("testIterativeEuclidean");
    printCanopies(canopies);
    verifyEuclideanCanopies(canopies);
  }

  /**
   * Story: User can produce initial canopy centers using a ManhattanDistanceMeasure and a CanopyMapper/Combiner which
   * clusters input points to produce an output set of canopy centroid points.
   */
  public void testCanopyMapperManhattan() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, new DummyReporter());
    }
    mapper.close();
    assertEquals("Number of map results", 1, collector.getData().size());
    // now verify the output
    List<VectorWritable> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error",
          manhattanCentroids.get(i).asFormatString(),
          data.get(i).get().asFormatString());
    }
  }

  /**
   * Story: User can produce initial canopy centers using a EuclideanDistanceMeasure and a CanopyMapper/Combiner which
   * clusters input points to produce an output set of canopy centroid points.
   */
  public void testCanopyMapperEuclidean() throws Exception {
    CanopyMapper mapper = new CanopyMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, new DummyReporter());
    }
    mapper.close();
    assertEquals("Number of map results", 1, collector.getData().size());
    // now verify the output
    List<VectorWritable> data = collector.getValue("centroid");
    assertEquals("Number of centroids", 3, data.size());
    for (int i = 0; i < data.size(); i++) {
      assertEquals("Centroid error",
          euclideanCentroids.get(i).asFormatString(),
          data.get(i).get().asFormatString());
    }
  }

  /**
   * Story: User can produce final canopy centers using a ManhattanDistanceMeasure and a CanopyReducer which clusters
   * input centroid points to produce an output set of final canopy centroid points.
   */
  public void testCanopyReducerManhattan() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    reducer.configure(conf);
    
    DummyOutputCollector<Text, Canopy> collector = new DummyOutputCollector<Text, Canopy>();
    List<VectorWritable> points = getPoints(raw);
    reducer.reduce(new Text("centroid"), points.iterator(), collector, null);
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Canopy> data = collector.getValue(key);
      assertEquals(manhattanCentroids.get(i).asFormatString() + " is not equal to " + data.get(0).computeCentroid().asFormatString(), manhattanCentroids.get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /**
   * Story: User can produce final canopy centers using a EuclideanDistanceMeasure and a CanopyReducer which clusters
   * input centroid points to produce an output set of final canopy centroid points.
   */
  public void testCanopyReducerEuclidean() throws Exception {
    CanopyReducer reducer = new CanopyReducer();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    reducer.configure(conf);
    
    DummyOutputCollector<Text, Canopy> collector = new DummyOutputCollector<Text, Canopy>();
    List<VectorWritable> points = getPoints(raw);
    reducer.reduce(new Text("centroid"), points.iterator(), collector, new DummyReporter());
    reducer.close();
    Set<String> keys = collector.getKeys();
    assertEquals("Number of centroids", 3, keys.size());
    int i = 0;
    for (String key : keys) {
      List<Canopy> data = collector.getValue(key);
      assertEquals(euclideanCentroids.get(i).asFormatString() + " is not equal to " + data.get(0).computeCentroid().asFormatString(), euclideanCentroids.get(i), data.get(0).computeCentroid());
      i++;
    }
  }

  /** Story: User can produce final canopy centers using a Hadoop map/reduce job and a ManhattanDistanceMeasure. */
  public void testCanopyGenManhattanMR() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    JobConf job = new JobConf(CanopyDriver.class);
    job.setMapOutputValueClass(points.get(0).getClass());
    ClusteringTestUtils.writePointsToFile(points, "testdata/file1", fs, job);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file2", fs, job);
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies", ManhattanDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    Path path = new Path("output/canopies/part-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Canopy canopy = new Canopy();
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("1st key", "C0", key.toString());
    //Canopy canopy = new Canopy(value);//Canopy.decodeCanopy(value.toString());
    assertEquals("1st x value", 1.5, canopy.getCenter().get(0));
    assertEquals("1st y value", 1.5, canopy.getCenter().get(1));
    assertTrue("more to come", reader.next(key, canopy));
    assertEquals("2nd key", "C1", key.toString());
    //canopy = Canopy.decodeCanopy(canopy.toString());
    assertEquals("1st x value", 4.333333333333334, canopy.getCenter().get(0));
    assertEquals("1st y value", 4.333333333333334, canopy.getCenter().get(1));
    assertFalse("more to come", reader.next(key, canopy));
    reader.close();
  }

  /** Story: User can produce final canopy centers using a Hadoop map/reduce job and a EuclideanDistanceMeasure. */
  public void testCanopyGenEuclideanMR() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    JobConf job = new JobConf(CanopyDriver.class);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file1", fs, job);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file2", fs, job);
    // now run the Canopy Driver
    CanopyDriver.runJob("testdata", "output/canopies", EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    Path path = new Path("output/canopies/part-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C0", key.toString());
    assertEquals("1st x value", 1.8, value.getCenter().get(0));
    assertEquals("1st y value", 1.8, value.getCenter().get(1));
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());
    assertEquals("1st x value", 4.433333333333334, value.getCenter().get(0));
    assertEquals("1st y value", 4.433333333333334, value.getCenter().get(1));
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }

  /** Story: User can cluster a subset of the points using a ClusterMapper and a ManhattanDistanceMeasure. */
  public void testClusterMapperManhattan() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    int nextCanopyId = 0;
    for (Vector centroid : manhattanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, new DummyReporter());
    }
    Map<String, List<VectorWritable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      String key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key, canopies);
      List<VectorWritable> pts = stringListEntry.getValue();
      for (VectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.get()));
      }
    }
  }

  private static Canopy findCanopy(String key, List<Canopy> canopies) {
    for (Canopy c : canopies) {
      if (c.getIdentifier().equals(key)) {
        return c;
      }
    }
    return null;
  }

  /** Story: User can cluster a subset of the points using a ClusterMapper and a EuclideanDistanceMeasure. */
  public void testClusterMapperEuclidean() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    int nextCanopyId = 0;
    for (Vector centroid : euclideanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, new DummyReporter());
    }
    Map<String, List<VectorWritable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      String key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key, canopies);
      List<VectorWritable> pts = stringListEntry.getValue();
      for (VectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.get()));
      }
    }
  }

  /** Story: User can cluster a subset of the points using a ClusterReducer and a ManhattanDistanceMeasure. */
  public void testClusterReducerManhattan() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.ManhattanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    int nextCanopyId = 0;
    for (Vector centroid : manhattanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, new DummyReporter());
    }
    Map<String, List<VectorWritable>> data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());

    // reduce the data
    Reducer<Text, VectorWritable, Text, VectorWritable> reducer = new IdentityReducer<Text, VectorWritable>();
    collector = new DummyOutputCollector<Text, VectorWritable>();
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      reducer.reduce(new Text(stringListEntry.getKey()), stringListEntry
          .getValue().iterator(), collector, null);
    }

    // check the output
    data = collector.getData();
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      String key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key, canopies);
      List<VectorWritable> pts = stringListEntry.getValue();
      for (VectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.get()));
      }
    }
  }

  /** Story: User can cluster a subset of the points using a ClusterReducer and a EuclideanDistanceMeasure. */
  public void testClusterReducerEuclidean() throws Exception {
    ClusterMapper mapper = new ClusterMapper();
    JobConf conf = new JobConf();
    conf.set(CanopyConfigKeys.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.EuclideanDistanceMeasure");
    conf.set(CanopyConfigKeys.T1_KEY, String.valueOf(3.1));
    conf.set(CanopyConfigKeys.T2_KEY, String.valueOf(2.1));
    mapper.configure(conf);
    
    List<Canopy> canopies = new ArrayList<Canopy>();
    DummyOutputCollector<Text, VectorWritable> collector = new DummyOutputCollector<Text, VectorWritable>();
    int nextCanopyId = 0;
    for (Vector centroid : euclideanCentroids) {
      canopies.add(new Canopy(centroid, nextCanopyId++));
    }
    mapper.config(canopies);
    List<VectorWritable> points = getPoints(raw);
    // map the data
    for (VectorWritable point : points) {
      mapper.map(new Text(), point, collector, null);
    }
    Map<String, List<VectorWritable>> data = collector.getData();

    // reduce the data
    Reducer<Text, VectorWritable, Text, VectorWritable> reducer = new IdentityReducer<Text, VectorWritable>();
    collector = new DummyOutputCollector<Text, VectorWritable>();
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      reducer.reduce(new Text(stringListEntry.getKey()),
          stringListEntry.getValue().iterator(), collector, null);
    }

    // check the output
    data = collector.getData();
    assertEquals("Number of map results", canopies.size(), data.size());
    for (Map.Entry<String, List<VectorWritable>> stringListEntry : data.entrySet()) {
      String key = stringListEntry.getKey();
      Canopy canopy = findCanopy(key, canopies);
      List<VectorWritable> pts = stringListEntry.getValue();
      for (VectorWritable ptDef : pts) {
        assertTrue("Point not in canopy", mapper.canopyCovers(canopy, ptDef.get()));
      }
    }
  }

  /** Story: User can produce final point clustering using a Hadoop map/reduce job and a ManhattanDistanceMeasure. */
  public void testClusteringManhattanMR() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    JobConf conf = new JobConf();
    conf.setMapOutputValueClass(points.get(0).getClass());
    ClusteringTestUtils.writePointsToFile(points, "testdata/file1", fs, conf);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file2", fs, conf);
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        ManhattanDistanceMeasure.class.getName(), 3.1, 2.1);
    //TODO: change
    Path path = new Path("output/clusters/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    int count = 0;
    /*while (reader.ready()) {
      System.out.println(reader.readLine());
      count++;
    }*/
    Text txt = new Text();
    VectorWritable vector = new VectorWritable();
    while (reader.next(txt, vector)) {
      count++;
      System.out.println("Txt: " + txt + " Vec: " + vector.get().asFormatString());
    }
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }

  /** Story: User can produce final point clustering using a Hadoop map/reduce job and a EuclideanDistanceMeasure. */
  public void testClusteringEuclideanMR() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, "testdata/file1", fs, conf);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file2", fs, conf);
    // now run the Job
    CanopyClusteringJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), 3.1, 2.1);
    Path path = new Path("output/clusters/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    int count = 0;
    /*while (reader.ready()) {
      System.out.println(reader.readLine());
      count++;
    }*/
    Text txt = new Text();
    VectorWritable can = new VectorWritable();
    while (reader.next(txt, can)) {
      count++;
    }
    /*while (reader.ready()) {
      System.out.println(reader.readLine());
      count++;
    }*/
    // the point [3.0,3.0] is covered by both canopies
    assertEquals("number of points", 2 + 2 * points.size(), count);
    reader.close();
  }


  /** Story: Clustering algorithm must support arbitrary user defined distance measure */
  public void testUserDefinedDistanceMeasure() throws Exception {
    List<VectorWritable> points = getPoints(raw);
    File testData = new File("testdata");
    if (!testData.exists()) {
      testData.mkdir();
    }
    Configuration conf = new Configuration();
    ClusteringTestUtils.writePointsToFile(points, "testdata/file1", fs, conf);
    ClusteringTestUtils.writePointsToFile(points, "testdata/file2", fs, conf);
    // now run the Canopy Driver. User defined measure happens to be a Manhattan
    // subclass so results are same.
    CanopyDriver.runJob("testdata", "output/canopies", UserDefinedDistanceMeasure.class.getName(), 3.1, 2.1);

    // verify output from sequence file
    JobConf job = new JobConf(CanopyDriver.class);
    Path path = new Path("output/canopies/part-00000");
    FileSystem fs = FileSystem.get(path.toUri(), job);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
    Text key = new Text();
    Canopy value = new Canopy();
    assertTrue("more to come", reader.next(key, value));
    assertEquals("1st key", "C0", key.toString());

    assertEquals("1st x value", 1.5, value.getCenter().get(0));
    assertEquals("1st y value", 1.5, value.getCenter().get(1));
    assertTrue("more to come", reader.next(key, value));
    assertEquals("2nd key", "C1", key.toString());

    assertEquals("1st x value", 4.333333333333334, value.getCenter().get(0));
    assertEquals("1st y value", 4.333333333333334, value.getCenter().get(1));
    assertFalse("more to come", reader.next(key, value));
    reader.close();
  }
}
