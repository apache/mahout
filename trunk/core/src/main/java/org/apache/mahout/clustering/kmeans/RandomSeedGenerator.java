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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Given an Input Path containing a {@link org.apache.hadoop.io.SequenceFile}, randomly select k vectors and
 * write them to the output file as a {@link org.apache.mahout.clustering.kmeans.Cluster} representing the
 * initial centroid to use.
 * <p/>
 */
public final class RandomSeedGenerator {
  
  private static final Logger log = LoggerFactory.getLogger(RandomSeedGenerator.class);
  
  public static final String K = "k";
  
  private RandomSeedGenerator() {
  }
  
  public static Path buildRandom(Path input, Path output, int k, DistanceMeasure measure) throws IOException,
                                                                    IllegalAccessException,
                                                                    InstantiationException {
    // delete the output directory
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    HadoopUtil.overwriteOutput(output);
    Path outFile = new Path(output, "part-randomSeed");
    boolean newFile = fs.createNewFile(outFile);
    if (newFile) {
      Path inputPathPattern;

      if (fs.getFileStatus(input).isDir()) {
        inputPathPattern = new Path(input, "*");
      } else {
        inputPathPattern = input;
      }
      
      FileStatus[] inputFiles = fs.globStatus(inputPathPattern);
      SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outFile, Text.class, Cluster.class);
      Random random = RandomUtils.getRandom();
      List<Text> chosenTexts = new ArrayList<Text>(k);
      List<Cluster> chosenClusters = new ArrayList<Cluster>(k);
      int nextClusterId = 0;
      
      for (FileStatus fileStatus : inputFiles) {
        if (fileStatus.isDir() || fileStatus.getPath().getName().startsWith("_")) {
          continue; // select only the top level files that do not begin with "_" (Cloudera CHD3 adds _SUCCESS file)
        }
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, fileStatus.getPath(), conf);
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        VectorWritable value = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        while (reader.next(key, value)) {
          Cluster newCluster = new Cluster(value.get(), nextClusterId++, measure);
          newCluster.observe(value.get(), 1);
          Text newText = new Text(key.toString());
          int currentSize = chosenTexts.size();
          if (currentSize < k) {
            chosenTexts.add(newText);
            chosenClusters.add(newCluster);
          } else if (random.nextInt(currentSize + 1) == 0) { // with chance 1/(currentSize+1) pick new element
            int indexToRemove = random.nextInt(currentSize); // evict one chosen randomly
            chosenTexts.remove(indexToRemove);
            chosenClusters.remove(indexToRemove);
            chosenTexts.add(newText);
            chosenClusters.add(newCluster);
          }
        }
        reader.close();
      }
      
      for (int i = 0; i < k; i++) {
        writer.append(chosenTexts.get(i), chosenClusters.get(i));
      }
      log.info("Wrote {} vectors to {}", k, outFile);
      writer.close();
    }
    
    return outFile;
  }
  
  public static List<Vector> chooseRandomPoints(Iterable<Vector> vectors, int k) {
    List<Vector> chosenPoints = new ArrayList<Vector>(k);
    Random random = RandomUtils.getRandom();
    for (Vector value : vectors) {
      int currentSize = chosenPoints.size();
      if (currentSize < k) {
        chosenPoints.add(value);
      } else if (random.nextInt(currentSize + 1) == 0) { // with chance 1/(currentSize+1) pick new element
        int indexToRemove = random.nextInt(currentSize); // evict one chosen randomly
        chosenPoints.remove(indexToRemove);
        chosenPoints.add(value);
      }
    }
    return chosenPoints;
  }
}
