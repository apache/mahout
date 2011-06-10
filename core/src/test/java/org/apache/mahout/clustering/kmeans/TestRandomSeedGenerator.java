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

import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public final class TestRandomSeedGenerator extends MahoutTestCase {
  
  private static final double[][] RAW = {{1, 1}, {2, 1}, {1, 2}, {2, 2},
    {3, 3}, {4, 4}, {5, 4}, {4, 5}, {5, 5}};
  
  private FileSystem fs;
  
  private static List<VectorWritable> getPoints() {
    List<VectorWritable> points = Lists.newArrayList();
    for (double[] fr : RAW) {
      Vector vec = new RandomAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }
  
  /** Story: test random seed generation generates 4 clusters with proper ids and data */
  @Test
  public void testRandomSeedGenerator() throws Exception {
    List<VectorWritable> points = getPoints();
    Job job = new Job();
    Configuration conf = job.getConfiguration();
    job.setMapOutputValueClass(VectorWritable.class);
    Path input = getTestTempFilePath("random-input");
    Path output = getTestTempDirPath("random-output");
    ClusteringTestUtils.writePointsToFile(points, input, fs, conf);
    
    RandomSeedGenerator.buildRandom(conf, input, output, 4, new ManhattanDistanceMeasure());

    int clusterCount = 0;
    Collection<Integer> set = new HashSet<Integer>();
    for (AbstractCluster value :
         new SequenceFileValueIterable<AbstractCluster>(new Path(output, "part-randomSeed"), true, conf)) {
      clusterCount++;
      int id = value.getId();
      assertTrue(set.add(id)); // validate unique id's
      
      Vector v = value.getCenter();
      assertVectorEquals(RAW[id], v); // validate values match
    }

    assertEquals(4, clusterCount); // validate sample count
  }
  
  private static void assertVectorEquals(double[] raw, Vector v) {
    assertEquals(raw.length, v.size());
    for (int i=0; i < raw.length; i++) {
      assertEquals(raw[i], v.getQuick(i), EPSILON);
    }
  }
}
