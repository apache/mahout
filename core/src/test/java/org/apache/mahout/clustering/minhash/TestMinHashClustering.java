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
package org.apache.mahout.clustering.minhash;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.minhash.HashFactory.HashType;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.commandline.MinhashOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class TestMinHashClustering extends MahoutTestCase {
  
  public static final double[][] REFERENCE = { {1, 2, 3, 4, 5}, {2, 1, 3, 6, 7}, {3, 7, 6, 11, 8, 9},
                                              {4, 7, 8, 9, 6, 1}, {5, 8, 10, 4, 1}, {6, 17, 14, 15},
                                              {8, 9, 11, 6, 12, 1, 7}, {10, 13, 9, 7, 4, 6, 3},
                                              {3, 5, 7, 9, 2, 11}, {13, 7, 6, 8, 5}};

  private Path input;
  private Path output;
  
  public static List<VectorWritable> getPointsWritable(double[][] raw) {
    List<VectorWritable> points = Lists.newArrayList();
    for (double[] fr : raw) {
      Vector vec = new SequentialAccessSparseVector(fr.length);
      vec.assign(fr);
      points.add(new VectorWritable(vec));
    }
    return points;
  }
  
  @Override
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    input = getTestTempDirPath("points");
    output = new Path(getTestTempDirPath(), "output");
    Path pointFile = new Path(input, "file1");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, pointFile, Text.class, VectorWritable.class);
    try {
      int id = 0;
      for (VectorWritable point : points) {
        writer.append(new Text("Id-" + id++), point);
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
  
  private String[] makeArguments(int minClusterSize,
                                 int minVectorSize,
                                 int numHashFunctions,
                                 int keyGroups,
                                 String hashType) {
    return new String[] {optKey(DefaultOptionCreator.INPUT_OPTION), input.toString(),
                         optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
                         optKey(MinhashOptionCreator.MIN_CLUSTER_SIZE), String.valueOf(minClusterSize),
                         optKey(MinhashOptionCreator.MIN_VECTOR_SIZE), String.valueOf(minVectorSize),
                         optKey(MinhashOptionCreator.HASH_TYPE), hashType,
                         optKey(MinhashOptionCreator.NUM_HASH_FUNCTIONS), String.valueOf(numHashFunctions),
                         optKey(MinhashOptionCreator.KEY_GROUPS), String.valueOf(keyGroups),
                         optKey(MinhashOptionCreator.NUM_REDUCERS), "1",
                         optKey(MinhashOptionCreator.DEBUG_OUTPUT), "true"};
  }
  
  private static Set<Integer> getValues(Vector vector) {
    Iterator<Vector.Element> itr = vector.iterator();
    Set<Integer> values = new HashSet<Integer>();
    while (itr.hasNext()) {
      values.add((int) itr.next().get());
    }
    return values;
  }
  
  private static void runPairwiseSimilarity(List<Vector> clusteredItems, double simThreshold, String msg) {
    if (clusteredItems.size() > 1) {
      for (int i = 0; i < clusteredItems.size(); i++) {
        Set<Integer> itemSet1 = getValues(clusteredItems.get(i));
        for (int j = i + 1; j < clusteredItems.size(); j++) {
          Set<Integer> itemSet2 = getValues(clusteredItems.get(j));
          Collection<Integer> union = new HashSet<Integer>();
          union.addAll(itemSet1);
          union.addAll(itemSet2);
          Collection<Integer> intersect = new HashSet<Integer>();
          intersect.addAll(itemSet1);
          intersect.retainAll(itemSet2);
          double similarity = intersect.size() / (double) union.size();
          assertTrue(msg + " - Sets failed min similarity test, Set1: " + itemSet1 + " Set2: " + itemSet2
                     + ", similarity:" + similarity, similarity >= simThreshold);
        }
      }
    }
  }
  
  private static void verify(Path output, double simThreshold, String msg) {
    Configuration conf = new Configuration();
    Path outputFile = new Path(output, "part-r-00000");
    List<Vector> clusteredItems = Lists.newArrayList();
    String prevClusterId = "";
    for (Pair<Writable,VectorWritable> record : new SequenceFileIterable<Writable,VectorWritable>(outputFile, conf)) {
      Writable clusterId = record.getFirst();
      VectorWritable point = record.getSecond();
      if (prevClusterId.equals(clusterId.toString())) {
        clusteredItems.add(point.get());
      } else {
        runPairwiseSimilarity(clusteredItems, simThreshold, msg);
        clusteredItems.clear();
        prevClusterId = clusterId.toString();
        clusteredItems.add(point.get());
      }
    }
    runPairwiseSimilarity(clusteredItems, simThreshold, msg);
  }
  
  @Test
  public void testLinearMinHashMRJob() throws Exception {
    String[] args = makeArguments(2, 3, 20, 3, HashType.LINEAR.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("Minhash MR Job failed for " + HashType.LINEAR, 0, ret);
    verify(output, 0.2, "Hash Type: LINEAR");
  }
  
  @Test
  public void testPolynomialMinHashMRJob() throws Exception {
    String[] args = makeArguments(2, 3, 20, 3, HashType.POLYNOMIAL.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("Minhash MR Job failed for " + HashType.POLYNOMIAL, 0, ret);
    verify(output, 0.3, "Hash Type: POLYNOMIAL");
  }
  
  @Test
  public void testMurmurMinHashMRJob() throws Exception {
    String[] args = makeArguments(2, 3, 20, 4, HashType.MURMUR.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("Minhash MR Job failed for " + HashType.MURMUR, 0, ret);
    verify(output, 0.3, "Hash Type: MURMUR");
  }
  
}