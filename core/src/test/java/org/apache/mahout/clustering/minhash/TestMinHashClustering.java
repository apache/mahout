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
import com.google.common.collect.Sets;
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
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Set;

@Deprecated
public final class TestMinHashClustering extends MahoutTestCase {
  
  private static final double[][] REFERENCE = { {0, 0, 3, 4, 5}, {0, 0, 3, 6, 7}, {0, 7, 6, 11, 8, 9},
                                              {0, 7, 8, 9, 6, 0}, {5, 8, 10, 0, 0}, {6, 17, 14, 15},
                                              {8, 9, 11, 0, 12, 0, 7}, {10, 13, 9, 7, 0, 6, 0},
                                              {0, 0, 7, 9, 0, 11}, {13, 7, 6, 8, 0}};

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
    Configuration conf = getConfiguration();
    List<VectorWritable> points = getPointsWritable(REFERENCE);
    input = getTestTempDirPath("points");
    output = new Path(getTestTempDirPath(), "output");
    Path pointFile = new Path(input, "file1");
    FileSystem fs = FileSystem.get(pointFile.toUri(), conf);
    SequenceFile.Writer writer = null;
    try {
      writer = new SequenceFile.Writer(fs, conf, pointFile, Text.class, VectorWritable.class);
      int id = 0;
      for (VectorWritable point : points) {
        writer.append(new Text("Id-" + id++), point);
      }
    } finally {
      Closeables.close(writer, false);
    }
  }
  
  private String[] makeArguments(String dimensionToHash, int minClusterSize,
                                 int minVectorSize,
                                 int numHashFunctions,
                                 int keyGroups,
                                 String hashType) {
    return new String[] {optKey(DefaultOptionCreator.INPUT_OPTION), input.toString(),
                         optKey(DefaultOptionCreator.OUTPUT_OPTION), output.toString(),
                         optKey(MinHashDriver.VECTOR_DIMENSION_TO_HASH), dimensionToHash,
                         optKey(MinHashDriver.MIN_CLUSTER_SIZE), String.valueOf(minClusterSize),
                         optKey(MinHashDriver.MIN_VECTOR_SIZE), String.valueOf(minVectorSize),
                         optKey(MinHashDriver.HASH_TYPE), hashType,
                         optKey(MinHashDriver.NUM_HASH_FUNCTIONS), String.valueOf(numHashFunctions),
                         optKey(MinHashDriver.KEY_GROUPS), String.valueOf(keyGroups),
                         optKey(MinHashDriver.NUM_REDUCERS), String.valueOf(1),
                         optKey(MinHashDriver.DEBUG_OUTPUT)};
  }
  
  private static Set<Integer> getValues(Vector vector, String dimensionToHash) {
    Set<Integer> values = Sets.newHashSet();
    if ("value".equalsIgnoreCase(dimensionToHash)) {
      for (Vector.Element e : vector.nonZeroes()) {
        values.add((int) e.get());
      }
    } else {
      for (Vector.Element e : vector.nonZeroes()) {
        values.add(e.index());
      }
    }
    return values;
  }

  private static void runPairwiseSimilarity(List<Vector> clusteredItems, double simThreshold,
                                            String dimensionToHash, String msg) {
    if (clusteredItems.size() > 1) {
      for (int i = 0; i < clusteredItems.size(); i++) {
        Set<Integer> itemSet1 = getValues(clusteredItems.get(i), dimensionToHash);
        for (int j = i + 1; j < clusteredItems.size(); j++) {
          Set<Integer> itemSet2 = getValues(clusteredItems.get(j), dimensionToHash);
          Collection<Integer> union = Sets.newHashSet();
          union.addAll(itemSet1);
          union.addAll(itemSet2);
          Collection<Integer> intersect = Sets.newHashSet();
          intersect.addAll(itemSet1);
          intersect.retainAll(itemSet2);
          double similarity = intersect.size() / (double) union.size();
          assertTrue(msg + " - Sets failed min similarity test, Set1: " + itemSet1 + " Set2: " + itemSet2
                     + ", similarity:" + similarity, similarity >= simThreshold);
        }
      }
    }
  }
  
  private void verify(Path output, double simThreshold, String dimensionToHash, String msg) throws IOException {
    Configuration conf = getConfiguration();
    Path outputFile = new Path(output, "part-r-00000");
    List<Vector> clusteredItems = Lists.newArrayList();
    String prevClusterId = "";
    for (Pair<Writable,VectorWritable> record : new SequenceFileIterable<Writable,VectorWritable>(outputFile, conf)) {
      Writable clusterId = record.getFirst();
      VectorWritable point = record.getSecond();
      if (prevClusterId.equals(clusterId.toString())) {
        clusteredItems.add(point.get());
      } else {
        runPairwiseSimilarity(clusteredItems, simThreshold, dimensionToHash, msg);
        clusteredItems.clear();
        prevClusterId = clusterId.toString();
        clusteredItems.add(point.get());
      }
    }
    runPairwiseSimilarity(clusteredItems, simThreshold, dimensionToHash, msg);
  }


  @Test
  public void testFailOnNonExistingHashType() throws Exception {
    String[] args = makeArguments("value", 2, 3, 20, 4, "xKrot37");
    int ret = ToolRunner.run(getConfiguration(), new MinHashDriver(), args);
    assertEquals(-1, ret);
  }

  @Test
  public void testLinearMinHashMRJob() throws Exception {
    String[] args = makeArguments("value", 2, 3, 20, 4, HashType.LINEAR.toString());
    int ret = ToolRunner.run(getConfiguration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Hash value Job failed for " + HashType.LINEAR, 0, ret);
    verify(output, 0.2, "value", "Hash Type: LINEAR");
  }
  
  @Test
  public void testPolynomialMinHashMRJob() throws Exception {
    String[] args = makeArguments("value", 2, 3, 20, 3, HashType.POLYNOMIAL.toString());
    int ret = ToolRunner.run(getConfiguration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash value failed for " + HashType.POLYNOMIAL, 0, ret);
    verify(output, 0.27, "value", "Hash Type: POLYNOMIAL");
  }
  
  @Test
  public void testMurmurMinHashMRJob() throws Exception {
    String[] args = makeArguments("value", 2, 3, 20, 4, HashType.MURMUR.toString());
    int ret = ToolRunner.run(getConfiguration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash value failed for " + HashType.MURMUR, 0, ret);
    verify(output, 0.2, "value", "Hash Type: MURMUR");
  }

  @Test
  public void testMurmur3MinHashMRJob() throws Exception {
    String[] args = makeArguments("value", 2, 3, 20, 4, HashType.MURMUR3.toString());
    int ret = ToolRunner.run(getConfiguration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash value failed for " + HashType.MURMUR3, 0, ret);
    verify(output, 0.2, "value", "Hash Type: MURMUR");
  }

  @Test
  public void testLinearMinHashMRJobHashIndex() throws Exception {
    String[] args = makeArguments("index", 2, 3, 20, 3, HashType.LINEAR.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash Index failed for " + HashType.LINEAR, 0, ret);
    verify(output, 0.2, "index", "Hash Type: LINEAR");
  }

  @Test
  public void testPolynomialMinHashMRJobHashIndex() throws Exception {
    String[] args = makeArguments("index", 2, 3, 20, 3, HashType.POLYNOMIAL.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash Index failed for " + HashType.POLYNOMIAL, 0, ret);
    verify(output, 0.3, "index", "Hash Type: POLYNOMIAL");
  }

  @Test
  public void testMurmurMinHashMRJobHashIndex() throws Exception {
    String[] args = makeArguments("index", 2, 3, 20, 4, HashType.MURMUR.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash Index failed for " + HashType.MURMUR, 0, ret);
    verify(output, 0.3, "index", "Hash Type: MURMUR");
  }

  @Test
  public void testMurmur3MinHashMRJobHashIndex() throws Exception {
    String[] args = makeArguments("index", 2, 3, 20, 4, HashType.MURMUR3.toString());
    int ret = ToolRunner.run(new Configuration(), new MinHashDriver(), args);
    assertEquals("MinHash MR Job Hash Index failed for " + HashType.MURMUR3, 0, ret);
    verify(output, 0.3, "index", "Hash Type: MURMUR");
  }

}