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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.Summarizable;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorBenchmarks implements Summarizable {

  private static final Logger log = LoggerFactory.getLogger(VectorBenchmarks.class);

  private final Vector[][] vectors;
  private final List<Vector> randomVectors = new ArrayList<Vector>();
  private final int cardinality;
  private final int numVectors;
  private final int loop;
  private final int opsPerUnit;
  private final Map<String,Integer> implType = new HashMap<String,Integer>();
  private final Map<String,List<String[]>> statsMap = new HashMap<String,List<String[]>>();
  
  public VectorBenchmarks(int cardinality, int numVectors, int loop, int opsPerUnit) {
    Random r = RandomUtils.getRandom();
    this.cardinality = cardinality;
    this.numVectors = numVectors;
    this.loop = loop;
    this.opsPerUnit = opsPerUnit;
    for (int i = 0; i < numVectors; i++) {
      Vector v = new DenseVector(cardinality);
      for (int j = 0; j < cardinality; j++) {
        double value = r.nextGaussian();
        v.set(j, value);
      }
      randomVectors.add(v);
    }
    vectors = new Vector[3][numVectors];
    
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
    float speed = multiplier * loop * numVectors * cardinality * 1000.0f * 12 / stats.getSumTime();
    float opsPerSec = loop * numVectors * 1000000000.0f / stats.getSumTime();
    log.info("{} {} \n{} {} \nSpeed: {} UnitsProcessed/sec {} MBytes/sec                                   ",
      new Object[] {benchmarkName, implName, content, stats.toString(), opsPerSec, speed});
    String info = stats.toString().replaceAll("\n", "\t") + "\tSpeed = " + opsPerSec + " /sec\tRate = "
                  + speed + " MB/s";
    if (implType.containsKey(implName) == false) {
      implType.put(implName, implType.size());
    }
    int implId = implType.get(implName);
    if (statsMap.containsKey(benchmarkName) == false) {
      statsMap.put(benchmarkName, new ArrayList<String[]>());
    }
    List<String[]> implStats = statsMap.get(benchmarkName);
    while (implStats.size() < implId + 1) {
      implStats.add(new String[] {});
    }
    implStats.set(implId, info.split("\t"));
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
    printStats(stats, "Create", "DenseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[1][i] = new RandomAccessSparseVector(randomVectors.get(i));
        call.end();
      }
    }
    printStats(stats, "Create", "RandomAccessSparseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[2][i] = new SequentialAccessSparseVector(randomVectors.get(i));
        call.end();
      }
    }
    printStats(stats, "Create", "SequentialAccessSparseVector");
    
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
    printStats(stats, "Clone", "RandomAccessSparseVector");
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        vectors[2][i] = vectors[2][i].clone();
        call.end();
      }
    }
    printStats(stats, "Clone", "SequentialAccessSparseVector");
    
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
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[1][i].dot(vectors[1][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "RandomAccessSparseVector", "sum = " + result + ' ');
    
    stats = new TimingStatistics();
    for (int l = 0; l < loop; l++) {
      for (int i = 0; i < numVectors; i++) {
        TimingStatistics.Call call = stats.newCall();
        result += vectors[2][i].dot(vectors[2][(i + 1) % numVectors]);
        call.end();
      }
    }
    // print result to prevent hotspot from eliminating deadcode
    printStats(stats, "DotProduct", "SequentialAccessSparseVector", "sum = " + result + ' ');
    
  }
  
  public void distanceMeasureBenchark(DistanceMeasure measure) {
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
    printStats(stats, measure.getClass().getName(), "RandomAccessSparseVector", "minDistance = " + result
                                                                                + ' ');
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
    printStats(stats, measure.getClass().getName(), "SequentialAccessSparseVector", "minDistance = " + result
                                                                                    + ' ');
    
  }
  
  public static void main(String[] args) {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option vectorSizeOpt = obuilder.withLongName("vectorSize").withRequired(false).withArgument(
      abuilder.withName("vs").withMinimum(1).withMaximum(1).create()).withDescription(
      "Cardinality of the vector. Default 1000").withShortName("vs").create();
    Option numVectorsOpt = obuilder.withLongName("numVectors").withRequired(false).withArgument(
      abuilder.withName("nv").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of Vectors to create. Default: 100").withShortName("nv").create();
    Option loopOpt = obuilder.withLongName("loop").withRequired(false).withArgument(
      abuilder.withName("loop").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of times to loop. Default: 200").withShortName("l").create();
    Option numOpsOpt = obuilder.withLongName("numOps").withRequired(false).withArgument(
      abuilder.withName("numOps").withMinimum(1).withMaximum(1).create()).withDescription(
      "Number of operations to do per timer. "
          + "E.g In distance measure, the distance is calculated numOps times"
          + " and the total time is measured. Default: 10").withShortName("no").create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(vectorSizeOpt).withOption(numVectorsOpt)
        .withOption(loopOpt).withOption(numOpsOpt).withOption(helpOpt).create();
    
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
      VectorBenchmarks mark = new VectorBenchmarks(cardinality, numVectors, loop, numOps);
      mark.createBenchmark();
      mark.cloneBenchmark();
      mark.dotBenchmark();
      mark.distanceMeasureBenchark(new CosineDistanceMeasure());
      mark.distanceMeasureBenchark(new SquaredEuclideanDistanceMeasure());
      mark.distanceMeasureBenchark(new EuclideanDistanceMeasure());
      mark.distanceMeasureBenchark(new ManhattanDistanceMeasure());
      mark.distanceMeasureBenchark(new TanimotoDistanceMeasure());
      
      log.info("\n{}", mark.summarize());
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
    }
    
  }
  
  @Override
  public String summarize() {
    int pad = 30;
    StringBuilder sb = new StringBuilder(1000);
    sb.append(StringUtils.rightPad("BenchMarks", pad));
    for (int i = 0; i < implType.size(); i++) {
      for (Entry<String,Integer> e : implType.entrySet()) {
        if (e.getValue() == i) {
          sb.append(StringUtils.rightPad(e.getKey(), pad));
          break;
        }
      }
    }
    sb.append('\n');
    List<String> keys = new ArrayList<String>(statsMap.keySet());
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
