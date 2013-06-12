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
import java.text.DecimalFormat;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang3.StringUtils;
import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFn;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.ChebyshevDistanceMeasure;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.MinkowskiDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class VectorBenchmarks {
  private static final int MAX_TIME_MS = 5000;
  private static final int LEAD_TIME_MS = 15000;
  public static final String CLUSTERS = "Clusters";
  public static final String CREATE_INCREMENTALLY = "Create (incrementally)";
  public static final String CREATE_COPY = "Create (copy)";

  public static final String DENSE_FN_SEQ = "Dense.fn(Seq)";
  public static final String RAND_FN_DENSE = "Rand.fn(Dense)";
  public static final String SEQ_FN_RAND = "Seq.fn(Rand)";
  public static final String RAND_FN_SEQ = "Rand.fn(Seq)";
  public static final String SEQ_FN_DENSE = "Seq.fn(Dense)";
  public static final String DENSE_FN_RAND = "Dense.fn(Rand)";
  public static final String SEQ_SPARSE_VECTOR = "SeqSparseVector";
  public static final String RAND_SPARSE_VECTOR = "RandSparseVector";
  public static final String DENSE_VECTOR = "DenseVector";

  private static final Logger log = LoggerFactory.getLogger(VectorBenchmarks.class);
  private static final Pattern TAB_NEWLINE_PATTERN = Pattern.compile("[\n\t]");
  private static final String[] EMPTY = new String[0];
  private static final DecimalFormat DF = new DecimalFormat("#.##");

  /* package private */
  final Vector[][] vectors;
  final Vector[] clusters;
  final int cardinality;
  final int numNonZeros;
  final int numVectors;
  final int numClusters;
  final int loop = Integer.MAX_VALUE;
  final int opsPerUnit;
  final long maxTimeUsec;
  final long leadTimeUsec;

  private final List<Vector> randomVectors = Lists.newArrayList();
  private final List<int[]> randomVectorIndices = Lists.newArrayList();
  private final List<double[]> randomVectorValues = Lists.newArrayList();
  private final Map<String, Integer> implType = Maps.newHashMap();
  private final Map<String, List<String[]>> statsMap = Maps.newHashMap();
  private final BenchmarkRunner runner;
  private final Random r = RandomUtils.getRandom();

  public VectorBenchmarks(int cardinality, int numNonZeros, int numVectors, int numClusters,
      int opsPerUnit) {
    runner = new BenchmarkRunner(LEAD_TIME_MS, MAX_TIME_MS);
    maxTimeUsec = TimeUnit.MILLISECONDS.toNanos(MAX_TIME_MS);
    leadTimeUsec = TimeUnit.MILLISECONDS.toNanos(LEAD_TIME_MS);

    this.cardinality = cardinality;
    this.numNonZeros = numNonZeros;
    this.numVectors = numVectors;
    this.numClusters = numClusters;
    this.opsPerUnit = opsPerUnit;

    setUpVectors(cardinality, numNonZeros, numVectors);

    vectors = new Vector[3][numVectors];
    clusters = new Vector[numClusters];
  }

  private void setUpVectors(int cardinality, int numNonZeros, int numVectors) {
    for (int i = 0; i < numVectors; i++) {
      Vector v = new SequentialAccessSparseVector(cardinality, numNonZeros); // sparsity!
      BitSet featureSpace = new BitSet(cardinality);
      int[] indexes = new int[numNonZeros];
      double[] values = new double[numNonZeros];
      int j = 0;
      while (j < numNonZeros) {
        double value = r.nextGaussian();
        int index = r.nextInt(cardinality);
        if (!featureSpace.get(index) && value != 0) {
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
  }

  void printStats(TimingStatistics stats, String benchmarkName, String implName, String content) {
    printStats(stats, benchmarkName, implName, content, 1);
  }

  void printStats(TimingStatistics stats, String benchmarkName, String implName) {
    printStats(stats, benchmarkName, implName, "", 1);
  }

  private void printStats(TimingStatistics stats, String benchmarkName, String implName,
      String content, int multiplier) {
    float speed = multiplier * stats.getNCalls() * (numNonZeros * 1000.0f * 12 / stats.getSumTime());
    float opsPerSec = stats.getNCalls() * 1000000000.0f / stats.getSumTime();
    log.info("{} {} \n{} {} \nOps    = {} Units/sec\nIOps   = {} MBytes/sec", benchmarkName,
        implName, content, stats.toString(), DF.format(opsPerSec), DF.format(speed));

    if (!implType.containsKey(implName)) {
      implType.put(implName, implType.size());
    }
    int implId = implType.get(implName);
    if (!statsMap.containsKey(benchmarkName)) {
      statsMap.put(benchmarkName, Lists.<String[]>newArrayList());
    }
    List<String[]> implStats = statsMap.get(benchmarkName);
    while (implStats.size() < implId + 1) {
      implStats.add(EMPTY);
    }
    implStats.set(
        implId,
        TAB_NEWLINE_PATTERN.split(stats + "\tSpeed  = " + DF.format(opsPerSec) + " /sec\tRate   = "
            + DF.format(speed) + " MB/s"));
  }

  public void createData() {
    for (int i = 0; i < Math.max(numVectors, numClusters); ++i) {
      vectors[0][vIndex(i)] = new DenseVector(randomVectors.get(vIndex(i)));
      vectors[1][vIndex(i)] = new RandomAccessSparseVector(randomVectors.get(vIndex(i)));
      vectors[2][vIndex(i)] = new SequentialAccessSparseVector(randomVectors.get(vIndex(i)));
      if (numClusters > 0) {
        clusters[cIndex(i)] = new RandomAccessSparseVector(randomVectors.get(vIndex(i)));
      }
    }
  }

  public void createBenchmark() {
    printStats(runner.benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        vectors[0][vIndex(i)] = new DenseVector(randomVectors.get(vIndex(i)));
        return depends(vectors[0][vIndex(i)]);
      }
    }), CREATE_COPY, DENSE_VECTOR);

    printStats(runner.benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        vectors[1][vIndex(i)] = new RandomAccessSparseVector(randomVectors.get(vIndex(i)));
        return depends(vectors[1][vIndex(i)]);
      }
    }), CREATE_COPY, RAND_SPARSE_VECTOR);

    printStats(runner.benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        vectors[2][vIndex(i)] = new SequentialAccessSparseVector(randomVectors.get(vIndex(i)));
        return depends(vectors[2][vIndex(i)]);
      }
    }), CREATE_COPY, SEQ_SPARSE_VECTOR);

    if (numClusters > 0) {
      printStats(runner.benchmark(new BenchmarkFn() {
        @Override
        public Boolean apply(Integer i) {
          clusters[cIndex(i)] = new RandomAccessSparseVector(randomVectors.get(vIndex(i)));
          return depends(clusters[cIndex(i)]);
        }
      }), CREATE_COPY, CLUSTERS);
    }
  }

  private boolean buildVectorIncrementally(TimingStatistics stats, int randomIndex, Vector v, boolean useSetQuick) {
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

    TimingStatistics.Call call = stats.newCall(leadTimeUsec);
    if (useSetQuick) {
      for (int i : permutation) {
        v.setQuick(indexes[i], values[i]);
      }
    } else {
      for (int i : permutation) {
        v.set(indexes[i], values[i]);
      }
    }
    return call.end(maxTimeUsec);
  }

  public void incrementalCreateBenchmark() {
    TimingStatistics stats = new TimingStatistics();
    for (int i = 0; i < loop; i++) {
      vectors[0][vIndex(i)] = new DenseVector(cardinality);
      if (buildVectorIncrementally(stats, vIndex(i), vectors[0][vIndex(i)], false)) {
        break;
      }
    }
    printStats(stats, CREATE_INCREMENTALLY, DENSE_VECTOR);

    stats = new TimingStatistics();
    for (int i = 0; i < loop; i++) {
      vectors[1][vIndex(i)] = new RandomAccessSparseVector(cardinality);
      if (buildVectorIncrementally(stats, vIndex(i), vectors[1][vIndex(i)], false)) {
        break;
      }
    }
    printStats(stats, CREATE_INCREMENTALLY, RAND_SPARSE_VECTOR);

    stats = new TimingStatistics();
    for (int i = 0; i < loop; i++) {
      vectors[2][vIndex(i)] = new SequentialAccessSparseVector(cardinality);
      if (buildVectorIncrementally(stats, vIndex(i), vectors[2][vIndex(i)], false)) {
        break;
      }
    }
    printStats(stats, CREATE_INCREMENTALLY, SEQ_SPARSE_VECTOR);

    if (numClusters > 0) {
      stats = new TimingStatistics();
      for (int i = 0; i < loop; i++) {
        clusters[cIndex(i)] = new RandomAccessSparseVector(cardinality);
        if (buildVectorIncrementally(stats, vIndex(i), clusters[cIndex(i)], false)) {
          break;
        }
      }
      printStats(stats, CREATE_INCREMENTALLY, CLUSTERS);
    }
  }

  public int vIndex(int i) {
    return i % numVectors;
  }

  public int cIndex(int i) {
    return i % numClusters;
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option vectorSizeOpt = obuilder
        .withLongName("vectorSize")
        .withRequired(false)
        .withArgument(abuilder.withName("vs").withDefault(1000000).create())
        .withDescription("Cardinality of the vector. Default: 1000000").withShortName("vs").create();
    Option numNonZeroOpt = obuilder
        .withLongName("numNonZero")
        .withRequired(false)
        .withArgument(abuilder.withName("nz").withDefault(1000).create())
        .withDescription("Size of the vector. Default: 1000").withShortName("nz").create();
    Option numVectorsOpt = obuilder
        .withLongName("numVectors")
        .withRequired(false)
        .withArgument(abuilder.withName("nv").withDefault(25).create())
        .withDescription("Number of Vectors to create. Default: 25").withShortName("nv").create();
    Option numClustersOpt = obuilder
        .withLongName("numClusters")
        .withRequired(false)
        .withArgument(abuilder.withName("nc").withDefault(0).create())
        .withDescription("Number of clusters to create. Set to non zero to run cluster benchmark. Default: 0")
        .withShortName("nc").create();
    Option numOpsOpt = obuilder
        .withLongName("numOps")
        .withRequired(false)
        .withArgument(abuilder.withName("numOps").withDefault(10).create())
        .withDescription(
            "Number of operations to do per timer. "
                + "E.g In distance measure, the distance is calculated numOps times"
                + " and the total time is measured. Default: 10").withShortName("no").create();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(vectorSizeOpt).withOption(numNonZeroOpt)
        .withOption(numVectorsOpt).withOption(numOpsOpt).withOption(numClustersOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelpWithGenericOptions(group);
        return;
      }

      int cardinality = 1000000;
      if (cmdLine.hasOption(vectorSizeOpt)) {
        cardinality = Integer.parseInt((String) cmdLine.getValue(vectorSizeOpt));

      }

      int numClusters = 0;
      if (cmdLine.hasOption(numClustersOpt)) {
        numClusters = Integer.parseInt((String) cmdLine.getValue(numClustersOpt));
      }

      int numNonZero = 1000;
      if (cmdLine.hasOption(numNonZeroOpt)) {
        numNonZero = Integer.parseInt((String) cmdLine.getValue(numNonZeroOpt));
      }

      int numVectors = 25;
      if (cmdLine.hasOption(numVectorsOpt)) {
        numVectors = Integer.parseInt((String) cmdLine.getValue(numVectorsOpt));

      }

      int numOps = 10;
      if (cmdLine.hasOption(numOpsOpt)) {
        numOps = Integer.parseInt((String) cmdLine.getValue(numOpsOpt));

      }
      VectorBenchmarks mark = new VectorBenchmarks(cardinality, numNonZero, numVectors, numClusters, numOps);
      runBenchmark(mark);

      // log.info("\n{}", mark);
      log.info("\n{}", mark.asCsvString());
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
    }
  }

  private static void runBenchmark(VectorBenchmarks mark) throws IOException {
    // Required to set up data.
    mark.createData();

    mark.createBenchmark();
    if (mark.cardinality < 200000) {
      // Too slow.
      mark.incrementalCreateBenchmark();
    }

    new CloneBenchmark(mark).benchmark();
    new DotBenchmark(mark).benchmark();
    new PlusBenchmark(mark).benchmark();
    new MinusBenchmark(mark).benchmark();
    new TimesBenchmark(mark).benchmark();
    new SerializationBenchmark(mark).benchmark();

    DistanceBenchmark distanceBenchmark = new DistanceBenchmark(mark);
    distanceBenchmark.benchmark(new CosineDistanceMeasure());
    distanceBenchmark.benchmark(new SquaredEuclideanDistanceMeasure());
    distanceBenchmark.benchmark(new EuclideanDistanceMeasure());
    distanceBenchmark.benchmark(new ManhattanDistanceMeasure());
    distanceBenchmark.benchmark(new TanimotoDistanceMeasure());
    distanceBenchmark.benchmark(new ChebyshevDistanceMeasure());
    distanceBenchmark.benchmark(new MinkowskiDistanceMeasure());

    if (mark.numClusters > 0) {
      ClosestCentroidBenchmark centroidBenchmark = new ClosestCentroidBenchmark(mark);
      centroidBenchmark.benchmark(new CosineDistanceMeasure());
      centroidBenchmark.benchmark(new SquaredEuclideanDistanceMeasure());
      centroidBenchmark.benchmark(new EuclideanDistanceMeasure());
      centroidBenchmark.benchmark(new ManhattanDistanceMeasure());
      centroidBenchmark.benchmark(new TanimotoDistanceMeasure());
      centroidBenchmark.benchmark(new ChebyshevDistanceMeasure());
      centroidBenchmark.benchmark(new MinkowskiDistanceMeasure());
    }
  }

  private String asCsvString() {
    List<String> keys = Lists.newArrayList(statsMap.keySet());
    Collections.sort(keys);
    Map<Integer,String> implMap = Maps.newHashMap();
    for (Entry<String,Integer> e : implType.entrySet()) {
      implMap.put(e.getValue(), e.getKey());
    }

    StringBuilder sb = new StringBuilder(1000);
    for (String benchmarkName : keys) {
      int i = 0;
      for (String[] stats : statsMap.get(benchmarkName)) {
        if (stats.length < 8) {
          continue;
        }
        sb.append(benchmarkName).append(',');
        sb.append(implMap.get(i++)).append(',');
        sb.append(stats[7].trim().split("=|/")[1].trim());
        sb.append('\n');
      }
    }
    sb.append('\n');
    return sb.toString();
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

  public BenchmarkRunner getRunner() {
    return runner;
  }
}
