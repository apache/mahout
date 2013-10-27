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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;

public final class TestEigenSeedGenerator extends MahoutTestCase {

  private
   static final double[][] RAW = {{1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0},
                                  {0, 1, 0}, {0, 0, 1}, {0, 0, 1}};

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
    Configuration conf = getConfiguration();
    fs = FileSystem.get(conf);
  }

  @Test
  public void testEigenSeedGenerator() throws Exception {
    List<VectorWritable> points = getPoints();
    Job job = new Job();
    Configuration conf = job.getConfiguration();
    job.setMapOutputValueClass(VectorWritable.class);
    Path input = getTestTempFilePath("eigen-input");
    Path output = getTestTempDirPath("eigen-output");
    ClusteringTestUtils.writePointsToFile(points, input, fs, conf);

    EigenSeedGenerator.buildFromEigens(conf, input, output, 3, new ManhattanDistanceMeasure());

    int clusterCount = 0;
    Collection<Integer> set = new HashSet<Integer>();
    Vector v[] = new Vector[3];
    for (ClusterWritable clusterWritable :
         new SequenceFileValueIterable<ClusterWritable>(
             new Path(output, "part-eigenSeed"), true, conf)) {
      Cluster cluster = clusterWritable.getValue();
      int id = cluster.getId();
      assertTrue(set.add(id)); // validate unique id's
      v[id] = cluster.getCenter();
      clusterCount++;
    }
    assertEquals(3, clusterCount); // validate sample count
    // validate pair-wise orthogonality
    assertEquals(0, v[0].dot(v[1]), 1E-10);
    assertEquals(0, v[1].dot(v[2]), 1E-10);
    assertEquals(0, v[0].dot(v[2]), 1E-10);
  }

}
