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

import junit.framework.TestCase;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.matrix.CardinalityException;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.DummyOutputCollector;
import org.apache.mahout.utils.EuclideanDistanceMeasure;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.nio.charset.Charset;

public class TestMeanShift extends TestCase {

  Vector[] raw = null;

  //DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();

  final DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

  public TestMeanShift(String name) {
    super(name);
  }

  /**
   * Print the canopies to the transcript
   * 
   * @param canopies a List<Canopy>
   */
  private static void printCanopies(List<MeanShiftCanopy> canopies) {
    for (MeanShiftCanopy canopy : canopies) {
      System.out.println(canopy.toString());
    }
  }

  /** 
   * Write the given points to the file within an enclosing MeanShiftCanopy
   * @param points a Vector[] of points
   * @param fileName the String file name
   * @param payload a String payload that goes with each point.
   * TODO: handle payloads associated with points. Currently they are ignored
   * @throws IOException
   */
  private static void writePointsToFileWithPayload(Vector[] points, String fileName,
      String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), Charset.forName("UTF-8")));
    for (Vector point : points) {
      output.write(new MeanShiftCanopy(point).toString());
      output.write(payload);
      output.write('\n');
    }
    output.flush();
    output.close();
  }

  /**
   * Recursively remove the contents of a directory
   * 
   * @param path
   * @throws Exception
   */
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

  /**
   * Print a graphical representation of the clustered image points as a 10x10
   * character mask
   * 
   * @param canopies
   */
  private static void printImage(List<MeanShiftCanopy> canopies) {
    char[][] out = new char[10][10];
    for (int i = 0; i < out.length; i++)
      for (int j = 0; j < out[0].length; j++)
        out[i][j] = ' ';
    for (MeanShiftCanopy canopy : canopies) {
      int ch = 'A' + canopy.getCanopyId() - 100;
      for (Vector pt : canopy.getBoundPoints())
        out[(int) pt.getQuick(0)][(int) pt.getQuick(1)] = (char) ch;
    }
    for (char[] anOut : out) {
      System.out.println(anOut);
    }
  }

  private List<MeanShiftCanopy> getInitialCanopies() {
    List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();
    for (Vector aRaw : raw) {
      canopies.add(new MeanShiftCanopy(aRaw));
    }
    return canopies;
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rmr("output");
    rmr("testdata");
    raw = new Vector[100];
    for (int i = 0; i < 10; i++)
      for (int j = 0; j < 10; j++) {
        int ix = i * 10 + j;
        Vector v = new DenseVector(3);
        v.setQuick(0, i);
        v.setQuick(1, j);
        if (i == j)
          v.setQuick(2, 9);
        else if (i + j == 9)
          v.setQuick(2, 4.5);
        raw[ix] = v;
      }
  }

  /**
   * Story: User can exercise the reference implementation to verify that the
   * test datapoints are clustered in a reasonable manner.
   * 
   * @throws CardinalityException
   */
  public void testReferenceImplementation() {
    MeanShiftCanopy.config(new EuclideanDistanceMeasure(), 4.0, 1.0, 0.5);
    List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();
    // add all points to the canopies
    for (Vector aRaw : raw) {
      MeanShiftCanopy.mergeCanopy(new MeanShiftCanopy(aRaw), canopies);
    }
    boolean done = false;
    int iter = 1;
    while (!done) {// shift canopies to their centroids
      done = true;
      List<MeanShiftCanopy> migratedCanopies = new ArrayList<MeanShiftCanopy>();
      for (MeanShiftCanopy canopy : canopies) {
        done = canopy.shiftToMean() && done;
        MeanShiftCanopy.mergeCanopy(canopy, migratedCanopies);
      }
      canopies = migratedCanopies;
      printCanopies(canopies);
      printImage(canopies);
      System.out.println(iter++);
    }
  }

  /**
   * Story: User can produce initial canopy centers using a
   * EuclideanDistanceMeasure and a CanopyMapper/Combiner which clusters input
   * points to produce an output set of canopies.
   * 
   * @throws Exception
   */
  public void testCanopyMapperEuclidean() throws Exception {
    MeanShiftCanopyMapper mapper = new MeanShiftCanopyMapper();
    MeanShiftCanopyCombiner combiner = new MeanShiftCanopyCombiner();
    DummyOutputCollector<Text,WritableComparable<?>> collector = new DummyOutputCollector<Text,WritableComparable<?>>();
    MeanShiftCanopy.config(euclideanDistanceMeasure, 4, 1, 0.5);
    // get the initial canopies
    List<MeanShiftCanopy> canopies = getInitialCanopies();
    // build the reference set
    List<MeanShiftCanopy> refCanopies = new ArrayList<MeanShiftCanopy>();
    for (Vector aRaw : raw) {
      MeanShiftCanopy.mergeCanopy(new MeanShiftCanopy(aRaw), refCanopies);
    }

    // map the data
    for (MeanShiftCanopy canopy : canopies) {
      mapper.map(new Text(), new Text(canopy.toString()), collector, null);
    }
    assertEquals("Number of map results", 100, collector.getData().size());
    // now combine the mapper output
    MeanShiftCanopy.config(euclideanDistanceMeasure, 4, 1, 0.5);
    Map<String, List<WritableComparable<?>>> mapData = collector.getData();
    collector = new DummyOutputCollector<Text,WritableComparable<?>>();
    for (Map.Entry<String, List<WritableComparable<?>>> stringListEntry : mapData.entrySet())
      combiner.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector,
          null);

    // now verify the output
    List<WritableComparable<?>> data = collector.getValue("canopy");
    assertEquals("Number of canopies", refCanopies.size(), data.size());
    // add all points to the reference canopies
    Map<String, MeanShiftCanopy> refCanopyMap = new HashMap<String, MeanShiftCanopy>();
    for (MeanShiftCanopy canopy : refCanopies) {
      canopy.shiftToMean();
      refCanopyMap.put(canopy.getIdentifier(), canopy);
    }
    // build a map of the combiner output
    Map<String, MeanShiftCanopy> canopyMap = new HashMap<String, MeanShiftCanopy>();
    for (WritableComparable<?> d : data) {
      MeanShiftCanopy dc = MeanShiftCanopy.decodeCanopy(d.toString());
      canopyMap.put(dc.getIdentifier(), dc);
    }
    // compare the maps
    for (Map.Entry<String, MeanShiftCanopy> stringMeanShiftCanopyEntry : refCanopyMap.entrySet()) {
      MeanShiftCanopy ref = stringMeanShiftCanopyEntry.getValue();

      MeanShiftCanopy canopy = canopyMap.get((ref.isConverged() ? "V" : "C")
          + (ref.getCanopyId() - raw.length));
      assertEquals("ids", ref.getCanopyId(), canopy.getCanopyId() + 100);
      assertEquals("centers(" + ref.getIdentifier() + ')', ref.getCenter()
          .asWritableComparable().toString(), canopy.getCenter()
          .asWritableComparable().toString());
      assertEquals("bound points", ref.getBoundPoints().size(), canopy
          .getBoundPoints().size());
    }
  }

  /**
   * Story: User can produce final canopy centers using a
   * EuclideanDistanceMeasure and a CanopyReducer which clusters input centroid
   * points to produce an output set of final canopy centroid points.
   * 
   * @throws Exception
   */
  public void testCanopyReducerEuclidean() throws Exception {
    MeanShiftCanopyMapper mapper = new MeanShiftCanopyMapper();
    MeanShiftCanopyCombiner combiner = new MeanShiftCanopyCombiner();
    MeanShiftCanopyReducer reducer = new MeanShiftCanopyReducer();
    DummyOutputCollector<Text,WritableComparable<?>> collector = new DummyOutputCollector<Text,WritableComparable<?>>();
    MeanShiftCanopy.config(euclideanDistanceMeasure, 4, 1, 0.5);
    // get the initial canopies
    List<MeanShiftCanopy> canopies = getInitialCanopies();
    // build the reference set
    List<MeanShiftCanopy> refCanopies = new ArrayList<MeanShiftCanopy>();
    for (Vector aRaw : raw) {
      MeanShiftCanopy.mergeCanopy(new MeanShiftCanopy(aRaw), refCanopies);
    }
    List<MeanShiftCanopy> refCanopies2 = new ArrayList<MeanShiftCanopy>();
    for (MeanShiftCanopy canopy : refCanopies) {
      canopy.shiftToMean();
    }
    for (MeanShiftCanopy canopy : refCanopies) {
      MeanShiftCanopy.mergeCanopy(canopy, refCanopies2);
    }
    for (MeanShiftCanopy canopy : refCanopies) {
      canopy.shiftToMean();
    }

    // map the data
    for (MeanShiftCanopy canopy : canopies) {
      mapper.map(new Text(), new Text(canopy.toString()), collector, null);
    }
    assertEquals("Number of map results", 100, collector.getData().size());
    // now combine the mapper output
    MeanShiftCanopy.config(euclideanDistanceMeasure, 4, 1, 0.5);
    Map<String, List<WritableComparable<?>>> mapData = collector.getData();
    collector = new DummyOutputCollector<Text,WritableComparable<?>>();
    for (Map.Entry<String, List<WritableComparable<?>>> stringListEntry : mapData.entrySet())
      combiner.reduce(new Text(stringListEntry.getKey()), stringListEntry.getValue().iterator(), collector,
          null);
    // now reduce the combiner output
    DummyOutputCollector<Text,WritableComparable<?>> collector2 = new DummyOutputCollector<Text,WritableComparable<?>>();
    reducer.reduce(new Text("canopy"), collector.getValue("canopy").iterator(),
        collector2, null);

    // now verify the output
    assertEquals("Number of canopies", refCanopies2.size(), collector2
        .getKeys().size());
    // add all points to the reference canopies
    Map<String, MeanShiftCanopy> refCanopyMap = new HashMap<String, MeanShiftCanopy>();
    for (MeanShiftCanopy canopy : refCanopies2) {
      refCanopyMap.put(canopy.getIdentifier(), canopy);
    }
    // compare the maps
    for (Map.Entry<String, MeanShiftCanopy> stringMeanShiftCanopyEntry : refCanopyMap.entrySet()) {
      MeanShiftCanopy ref = stringMeanShiftCanopyEntry.getValue();

      List<WritableComparable<?>> values = collector2
          .getValue((ref.isConverged() ? "V" : "C")
              + (ref.getCanopyId() - raw.length));
      assertEquals("values", 1, values.size());
      MeanShiftCanopy canopy = MeanShiftCanopy.decodeCanopy(values.get(0)
          .toString());
      assertEquals("ids", ref.getCanopyId(), canopy.getCanopyId() + 100);
      assertEquals("centers(" + stringMeanShiftCanopyEntry.getKey() + ')', ref.getCenter()
          .asWritableComparable().toString(), canopy.getCenter()
          .asWritableComparable().toString());
      assertEquals("bound points", ref.getBoundPoints().size(), canopy
          .getBoundPoints().size());
    }
  }

  /**
   * Story: User can produce final point clustering using a Hadoop map/reduce
   * job and a EuclideanDistanceMeasure.
   * 
   * @throws Exception
   */
  public void testCanopyEuclideanMRJob() throws Exception {
    File testData = new File("testdata");
    if (!testData.exists())
      testData.mkdir();
    writePointsToFileWithPayload(raw, "testdata/file1", "");
    writePointsToFileWithPayload(raw, "testdata/file2", "");
    // now run the Job
    MeanShiftCanopyJob.runJob("testdata", "output",
        EuclideanDistanceMeasure.class.getName(), 4, 1, 0.5, 10);
    JobConf conf = new JobConf(MeanShiftCanopyDriver.class);
    FileSystem fs = FileSystem.get(conf);
    Path outPart = new Path("output/canopies-2/part-00000");
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, outPart, conf);
    Text key = new Text();
    Text value = new Text();
    int count = 0;
    while (reader.next(key, value)) {
      MeanShiftCanopy.decodeCanopy(value.toString());
      count++;
    }
    reader.close();
    assertEquals("count", 3, count);
  }
}
