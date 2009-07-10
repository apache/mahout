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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.matrix.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Given an Input Path containing a {@link org.apache.hadoop.io.SequenceFile}, randomly select k vectors and write them
 * to the output file as a {@link org.apache.mahout.clustering.kmeans.Cluster} representing the initial centroid to use.
 * <p/>
 */
public final class RandomSeedGenerator {

  private static final Logger log = LoggerFactory.getLogger(RandomSeedGenerator.class);

  public static final String K = "k";

  private RandomSeedGenerator() {
  }

  public static Path buildRandom(String input, String output,
                                 int k) throws IOException, IllegalAccessException, InstantiationException {
    // delete the output directory
    JobConf conf = new JobConf(RandomSeedGenerator.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (fs.exists(outPath)) {
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);
    Path outFile = new Path(outPath, "part-randomSeed");
    if (fs.exists(outFile) == true) {
      log.warn("Deleting " + outFile);
      fs.delete(outFile, false);
    }
    boolean newFile = fs.createNewFile(outFile);
    if (newFile == true) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(input), conf);
      Writable key = (Writable) reader.getKeyClass().newInstance();
      Vector value = (Vector) reader.getValueClass().newInstance();
      SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outFile, Text.class, Cluster.class);
      Random random = new Random();

      List<Text> chosenTexts = new ArrayList<Text>(k);
      List<Cluster> chosenClusters = new ArrayList<Cluster>(k);
      while (reader.next(key, value)) {
        Cluster newCluster = new Cluster(value);
        newCluster.addPoint(value);
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
      for (int i = 0; i < k; i++) {
        writer.append(chosenTexts.get(i), chosenClusters.get(i));
      }
      log.info("Wrote " + k + " vectors to " + outFile);
      reader.close();
      writer.close();
    }

    return outFile;
  }
}
