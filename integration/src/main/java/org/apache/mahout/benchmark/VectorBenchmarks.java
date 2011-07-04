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

package org.apache.mahout.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorBenchmarks {

  private static final Logger log = LoggerFactory.getLogger(VectorBenchmarks.class);

  private static final Pattern TAB_NEWLINE_PATTERN = Pattern.compile("[\n\t]");
  private static final String[] EMPTY = new String[0];

  private final Vector[][] vectors;
  private final Vector[] clusters;
  private final SparseMatrix clusterDistances;
  private final List<Vector> randomVectors = Lists.newArrayList();
  private final List<int[]> randomVectorIndices = Lists.newArrayList();
  private final List<double[]> randomVectorValues = Lists.newArrayList();
  private final int cardinality;
  private final int sparsity;
  private final int numVectors;
  private final int loop;
  private final int opsPerUnit;
  private final Map<String,Integer> implType = Maps.newHashMap();
  private final Map<String,List<String[]>> statsMap = Maps.newHashMap();
  private final int numClusters;
  
  public VectorBenchmarks(int cardinality, int sparsity, int numVectors, int numClusters, int loop, int opsPerUnit) {
    Random r = RandomUtils.getRandom();
    this.cardinality = cardinality;
    this.sparsity = sparsity;
    this.numVectors = numVectors;
    this.numClusters = numClusters;
    this.loop = loop;
    this.opsPerUnit = opsPerUnit;
    for (int i = 0; i < numVectors; i++) {
      Vector v = new SequentialAccessSparseVector(cardinality, sparsity); // sparsity!
      BitSet featureSpace = new BitSet(cardinality);
      int[] indexes = new int[sparsity];
      double[] values = new double[sparsity];
      int j = 0;
      while (j < sparsity) {
        double value = r.nextGaussian();
        int index = r.nextInt(cardinality);
        if (!featureSpace.get(index)) {
          featureSpace.set(index);
          indexes[j] = index;
          values[j++] = value;
          v.set(index, value);
        }
      }
      randomVectorIndices.add(indexes);
      randomVectorValues.add(values);
      randomVectors.add(v);
    }
    vectors = new Vector[3][numVectors];
    clusters = new Vector[numClusters];
    clusterDistances = new SparseMatrix(numClusters, numClusters);
  }
  
  private void printStats(TimingStatistics stats, String benchmarkName, String implName, String content) {
    printStats(stats, benchmarkName, implName, content, 1);
  }
  
  private void printStats(TimingStatistics stats, String benchmarkName, String implName) {
    printStats(stats, benchmarkName, implName, "", 1);
  }
  
  private void printStats(TimingStatistics stats,
                          String benchmarkName,
                          String implName,
                          String content,
                          int multiplier) {
    float speed = multiplier * loop * numVectors * sparsity * 1000.0f * 12 / stats.getSumTime();
    float opsPerSec = loop * numVectors * 1000000000.0f / stats.getSumTime();
    log.info("{} {} \n{} {} \nSpeed: {} UnitsProcessed/sec {} MBytes/sec                                   ",
      new Object[] {benchmarkName, implName, content, stats.toString(), opsPerSec, speed});

    if (!implType.containsKey(implName)) {
      implType.put(implName, implType.size());
    }
    int implId = implType.get(implName);
    if (!statsMap.containsKey(benchmarkName)) {
      statsMap.put(benchmarkName, new ArrayList<String[]>());
    }
    List<String[]> implStats = statsMap.get(benchmarkName);
    while (implStats.size() < implId + 1) {
      implStats.add(EMPTY);
    }
    implStats.set(implId,
                  TAB_NEWLINE_PATTERN.split(stats + "\tSpeed = " + opsPerSec + " /sec\tRate = " + speed + " MB/s"));
  }
  
  public void createBenchmark() {
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[0][i] = new DenseVector(randomVectors.get(i));
        call.end();
      }
    }
    printStats(stats, "Create (copy)", "DenseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[1][i] = new RandomAccessSparseVector(randomVectors.get(i));
        call.end();
      }
    }
    printStats(stats, "Create (copy)", "RandSparseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[2][i] = new SequentialAccessSparseVector(randomVectors.get(i));
        call.end();
      }
    }
    printStats(stats, "Create (copy)", "SeqSparseVector");
    
  }

  private void buildVectorIncrementally(TimingStatistics stats, int randomIndex, Vector v, boolean useSetQuick) {
    int[] indexes = randomVectorIndices.get(randomIndex);
    double[] values = randomVectorValues.get(randomIndex);
    List<Integer> randomOrder = Lists.newArrayList();
    for (int i = 0; i < indexes.length; i++) {
      randomOrder.add(i);
    }
    Collections.shuffle(randomOrder);
    int[] permutation = new int[randomOrder.size()];
    for (int i = 0; i < randomOrder.size(); i++) {
      permutation[i] = randomOrder.get(i);
    }

    TimingStatistics.Call call = stats.newCall();
    if (useSetQuick) {
      for (int i : permutation) {
        v.setQuick(indexes[i], values[i]);
      }
    } else {
      for (int i : permutation) {
        v.set(indexes[i], values[i]);
      }
    }
    call.end();
  }

  public void incrementalCreateBenchmark() {
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        vectors[0][i] = new DenseVector(cardinality);
        buildVectorIncrementally(stats, i, vectors[0][i], false);
      }
    }
    printStats(stats, "Create (incrementally)", "DenseVector");

    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        vectors[1][i] = new RandomAccessSparseVector(cardinality);
        buildVectorIncrementally(stats, i, vectors[1][i], false);
      }
    }
    printStats(stats, "Create (incrementally)", "RandSparseVector");

//    stats = new TimingStatistics();
//    for (int l = 0; l < loop; l++) {
//      for (int i = 0; i < numVectors; i++) {
//        vectors[2][i] = new SequentialAccessSparseVector(cardinality);
//        buildVectorIncrementally(stats, i, vectors[2][i], false);
//      }
//    }
//    printStats(stats, "Create (incrementally)", "SeqSparseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numClusters; i++) {
        clusters[i] = new RandomAccessSparseVector(cardinality);
        buildVectorIncrementally(stats, i, clusters[i], false);
      }
    }
    printStats(stats, "Create (incrementally)", "Clusters");
  }
  
  public void cloneBenchmark() {
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[0][i] = vectors[0][i].clone();
        call.end();
      }
    }
    printStats(stats, "Clone", "DenseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[1][i] = vectors[1][i].clone();
        call.end();
      }
    }
    printStats(stats, "Clone", "RandSparseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[2][i] = vectors[2][i].clone();
        call.end();
      }
    }
    printStats(stats, "Clone", "SeqSparseVector");
    
  }
  
  public void serializeBenchmark() throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
      new Path("/tmp/dense-vector"), IntWritable.class, VectorWritable.class);

    Writable one = new IntWritable(0);
    VectorWritable vec = new VectorWritable();
    TimingStatistics stats = new TimingStatistics();

    try {
      for (int l = 0; l < loop; l++) {
        for (int i = 0; i < numVectors; i++) {
          TimingStatistics.Call call = stats.newCall();
          vec.set(vectors[0][i]);
          writer.append(one, vec);
          call.end();
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    printStats(stats, "Serialize", "DenseVector");
    
    writer = new SequenceFile.Writer(fs, conf,
      new Path("/tmp/randsparse-vector"), IntWritable.class, VectorWritable.class);
    stats = new TimingStatistics();
    try {
      for (int l = 0; l < loop; l++) {
        for (int i = 0; i < numVectors; i++) {
          TimingStatistics.Call call = stats.newCall();
          vec.set(vectors[1][i]);
          writer.append(one, vec);
          call.end();
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    printStats(stats, "Serialize", "RandSparseVector");
    
    writer = new SequenceFile.Writer(fs, conf,
      new Path("/tmp/seqsparse-vector"), IntWritable.class, VectorWritable.class);
    stats = new TimingStatistics();
    try {
      for (int l = 0; l < loop; l++) {
        for (int i = 0; i < numVectors; i++) {
          TimingStatistics.Call call = stats.newCall();
          vec.set(vectors[2][i]);
          writer.append(one, vec);
          call.end();
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    printStats(stats, "Serialize", "SeqSparseVector");
    
  }
  
  public void deserializeBenchmark() throws IOException {
    doDeserializeBenchmark("DenseVector", "/tmp/dense-vector");
    doDeserializeBenchmark("RandSparseVector", "/tmp/randsparse-vector");
    doDeserializeBenchmark("SeqSparseVector", "/tmp/seqsparse-vector");
  }

  private void doDeserializeBenchmark(String name, String pathString) throws IOException {
    TimingStatistics stats = new TimingStatistics();
    TimingStatistics.Call call = stats.newCall();
    Iterator<?> iterator = new SequenceFileValueIterator<Writable>(new Path(pathString), true, new Configuration());
    while (iterator.hasNext()) {
      iterator.next();
      call.end();
      call = stats.newCall();
    }
    printStats(stats, "Deserialize", name);
  }
  
  public void dotBenchmark() {
    double result = 0;
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[0][i].dot(vectors[0][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "DenseVector", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[1][i].dot(vectors[1][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "RandSparseVector", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[2][i].dot(vectors[2][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "SeqSparseVector", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[0][i].dot(vectors[1][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Dense.fn(Rand)", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[0][i].dot(vectors[2][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Dense.fn(Seq)", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[1][i].dot(vectors[0][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Rand.fn(Dense)", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[1][i].dot(vectors[2][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Rand.fn(Seq)", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[2][i].dot(vectors[0][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Seq.fn(Dense)", "sum = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[2][i].dot(vectors[1][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "Seq.fn(Rand)", "sum = " + result + ' ');


  }


  public void closestCentroidBenchmark(DistanceMeasure measure) {

    for (int i = 0; i < numClusters; i++) {
      for (int j = 0; j < numClusters; j++) {
        double distance = Double.POSITIVE_INFINITY;
        if (i != j) {
          distance = measure.distance(clusters[i], clusters[j]);
        }
        clusterDistances.setQuick(i, j, distance);
      }
    }

    long distanceCalculations = 0;
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      TimingStatistics.Call call = stats.newCall();
      for (int i = 0; i < numVectors; i++) {
        Vector vector = vectors[1][i];
        double minDistance = Double.MAX_VALUE;
        for (int k = 0; k < numClusters; k++) {
          double distance = measure.distance(vector, clusters[k]);
          distanceCalculations++;
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
      }
      call.end();
    }
    printStats(stats,
               measure.getClass().getName(),
               "Closest center without Elkan's trick",
               "distanceCalculations = " + distanceCalculations);


    distanceCalculations = 0;
    stats = new TimingStatistics();
    Random rand = RandomUtils.getRandom();
    //rand.setSeed(System.currentTimeMillis());
    for (int l = 0; l < loop; l++) {
      TimingStatistics.Call call = stats.newCall();
      for (int i = 0; i < numVectors; i++) {
        Vector vector = vectors[1][i];
        int closestCentroid = rand.nextInt(numClusters);
        double dist = measure.distance(vector, clusters[closestCentroid]);
        distanceCalculations++;
        for (int k = 0; k < numClusters; k++) {
          if (closestCentroid != k) {
            double centroidDist = clusterDistances.getQuick(k, closestCentroid);
            if (centroidDist < 2 * dist) {
              dist = measure.distance(vector, clusters[k]);
              closestCentroid = k;
              distanceCalculations++;
            }
          }
        }
      }
      call.end();
    }
    printStats(stats,
               measure.getClass().getName(),
               "Closest center with Elkan's trick",
               "distanceCalculations = " + distanceCalculations);
  }

  public void distanceMeasureBenchmark(DistanceMeasure measure) {
    double result = 0;
    TimingStatistics stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[0][i], vectors[0][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "DenseVector", "minDistance = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[1][i], vectors[1][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "RandSparseVector", "minDistance = " + result
                                                                                + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[2][i], vectors[2][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "SeqSparseVector", "minDistance = " + result
                                                                                    + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[0][i], vectors[1][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Dense.fn(Rand)", "minDistance = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[0][i], vectors[2][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Dense.fn(Seq)", "minDistance = " + result
                                                                                + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[1][i], vectors[0][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Rand.fn(Dense)", "minDistance = " + result
                                                                                    + ' ');
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[1][i], vectors[2][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Rand.fn(Seq)", "minDistance = " + result + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[2][i], vectors[0][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Seq.fn(Dense)", "minDistance = " + result
                                                                                + ' ');
    result = 0;
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        double minDistance = Double.MAX_VALUE;
        for (int u = 0; u < opsPerUnit; u++) {
          double distance = measure.distance(vectors[2][i], vectors[1][u]);
          if (distance < minDistance) {
            minDistance = distance;
          }
        }
        result += minDistance;
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, measure.getClass().getName(), "Seq.fn(Rand)", "minDistance = " + result
                                                                                    + ' ');
    
  }
  
  public static void main(String[] args) throws IOException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option vectorSizeOpt = obuilder.withLongName("vectorSize").withRequired(false).withArgument(
      abuilder.withName("vs").withMinimum(1).withMaximum(1).create()).withDescription(
      "Cardinality of the vector. Default 1000").withShortName("vs").create();
    
    Option vectorSparsityOpt = obuilder.withLongName("sparsity").withRequired(false).withArgument(
      abuilder.withName("sp").withMinimum(1).withMaximum(1).create()).withDescription(
      "Sparsity of the vector. Default 1000").withShortName("sp").create();
    Option numVectorsOpt = obuilder.withLongName("numVectors").withRequired(false).withArgument(
      abuilder.withName("nv").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of Vectors to create. Default: 100").withShortName("nv").create();
    Option numClustersOpt = obuilder.withLongName("numClusters").withRequired(false).withArgument(
          abuilder.withName("vs").withMinimum(1).withMaximum(1).create()).withDescription(
          "Number of Vectors to create. Default: 10").withShortName("vs").create();
    Option loopOpt = obuilder.withLongName("loop").withRequired(false).withArgument(
      abuilder.withName("loop").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of times to loop. Default: 200").withShortName("l").create();
    Option numOpsOpt = obuilder.withLongName("numOps").withRequired(false).withArgument(
      abuilder.withName("numOps").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of operations to do per timer. "
          + "E.g In distance measure, the distance is calculated numOps times"
          + " and the total time is measured. Default: 10").withShortName("no").create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(vectorSizeOpt).withOption(vectorSparsityOpt)
        .withOption(numVectorsOpt).withOption(loopOpt).withOption(numOpsOpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      int cardinality = 1000;
      if (cmdLine.hasOption(vectorSizeOpt)) {
        cardinality = Integer.parseInt((String) cmdLine.getValue(vectorSizeOpt));
        
      }    
      
      int numClusters=25;
      if (cmdLine.hasOption(numClustersOpt)) {
        numClusters = Integer.parseInt((String) cmdLine.getValue(numClustersOpt));
      }

      int sparsity = 1000;
      if (cmdLine.hasOption(vectorSparsityOpt)) {
        sparsity = Integer.parseInt((String) cmdLine.getValue(vectorSparsityOpt));
      }

      int numVectors = 100;
      if (cmdLine.hasOption(numVectorsOpt)) {
        numVectors = Integer.parseInt((String) cmdLine.getValue(numVectorsOpt));
        
      }
      int loop = 200;
      if (cmdLine.hasOption(loopOpt)) {
        loop = Integer.parseInt((String) cmdLine.getValue(loopOpt));
        
      }
      int numOps = 10;
      if (cmdLine.hasOption(numOpsOpt)) {
        numOps = Integer.parseInt((String) cmdLine.getValue(numOpsOpt));
        
      }
      VectorBenchmarks mark = new VectorBenchmarks(cardinality, sparsity, numVectors, numClusters, loop, numOps);
      mark.createBenchmark();
      mark.incrementalCreateBenchmark();
      mark.cloneBenchmark();
      mark.dotBenchmark();
      mark.serializeBenchmark();
      mark.deserializeBenchmark();
      mark.distanceMeasureBenchmark(new CosineDistanceMeasure());
      mark.distanceMeasureBenchmark(new SquaredEuclideanDistanceMeasure());
      mark.distanceMeasureBenchmark(new EuclideanDistanceMeasure());
      mark.distanceMeasureBenchmark(new ManhattanDistanceMeasure());
      mark.distanceMeasureBenchmark(new TanimotoDistanceMeasure());
      
      mark.closestCentroidBenchmark(new CosineDistanceMeasure());
      mark.closestCentroidBenchmark(new SquaredEuclideanDistanceMeasure());
      mark.closestCentroidBenchmark(new EuclideanDistanceMeasure());
      mark.closestCentroidBenchmark(new ManhattanDistanceMeasure());
      mark.closestCentroidBenchmark(new TanimotoDistanceMeasure());
      
      log.info("\n{}", mark);
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
    }
    
  }
  
  @Override
  public String toString() {
    int pad = 24;
    StringBuilder sb = new StringBuilder(1000);
    sb.append(StringUtils.rightPad("BenchMarks", pad));
    for (int i = 0; i < implType.size(); i++) {
      for (Entry<String,Integer> e : implType.entrySet()) {
        if (e.getValue() == i) {
          sb.append(StringUtils.rightPad(e.getKey(), pad).substring(0, pad));
          break;
        }
      }
    }
    sb.append('\n');
    List<String> keys = Lists.newArrayList(statsMap.keySet());
    Collections.sort(keys);
    for (String benchmarkName : keys) {
      List<String[]> implTokenizedStats = statsMap.get(benchmarkName);
      int maxStats = 0;
      for (String[] stat : implTokenizedStats) {
        maxStats = Math.max(maxStats, stat.length);
      }
      
      for (int i = 0; i < maxStats; i++) {
        boolean printedName = false;
        for (String[] stats : implTokenizedStats) {
          if (i == 0 && !printedName) {
            sb.append(StringUtils.rightPad(benchmarkName, pad));
            printedName = true;
          } else if (!printedName) {
            printedName = true;
            sb.append(StringUtils.rightPad("", pad));
          }
          if (stats.length > i) {
            sb.append(StringUtils.rightPad(stats[i], pad));
          } else {
            sb.append(StringUtils.rightPad("", pad));
          }

        }
        sb.append('\n');
      }
      sb.append('\n');
    }
    return sb.toString();
  }
  
}