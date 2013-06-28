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
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

/**
 * Given an Input Path containing a {@link org.apache.hadoop.io.SequenceFile}, select k vectors and write them to the
 * output file as a {@link org.apache.mahout.clustering.kmeans.Kluster} representing the initial centroid to use. The
 * selection criterion is the rows with max value in that respective column
 */
public final class EigenSeedGenerator {

  private static final Logger log = LoggerFactory.getLogger(EigenSeedGenerator.class);

  public static final String K = "k";

  private EigenSeedGenerator() {}

  public static Path buildFromEigens(Configuration conf, Path input, Path output, int k, DistanceMeasure measure)
      throws IOException {
    // delete the output directory
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    HadoopUtil.delete(conf, output);
    Path outFile = new Path(output, "part-eigenSeed");
    boolean newFile = fs.createNewFile(outFile);
    if (newFile) {
      Path inputPathPattern;

      if (fs.getFileStatus(input).isDir()) {
        inputPathPattern = new Path(input, "*");
      } else {
        inputPathPattern = input;
      }

      FileStatus[] inputFiles = fs.globStatus(inputPathPattern, PathFilters.logsCRCFilter());
      SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outFile, Text.class, ClusterWritable.class);
      Map<Integer,Double> maxEigens = Maps.newHashMapWithExpectedSize(k); // store
                                                                          // max
                                                                          // value
                                                                          // of
                                                                          // each
                                                                          // column
      Map<Integer,Text> chosenTexts = Maps.newHashMapWithExpectedSize(k);
      Map<Integer,ClusterWritable> chosenClusters = Maps.newHashMapWithExpectedSize(k);

      for (FileStatus fileStatus : inputFiles) {
        if (!fileStatus.isDir()) {
          for (Pair<Writable,VectorWritable> record : new SequenceFileIterable<Writable,VectorWritable>(
              fileStatus.getPath(), true, conf)) {
            Writable key = record.getFirst();
            VectorWritable value = record.getSecond();

            for (Vector.Element e : value.get().nonZeroes()) {
              int index = e.index();
              double v = Math.abs(e.get());

              if (!maxEigens.containsKey(index) || v > maxEigens.get(index)) {
                maxEigens.put(index, v);
                Text newText = new Text(key.toString());
                chosenTexts.put(index, newText);
                Kluster newCluster = new Kluster(value.get(), index, measure);
                newCluster.observe(value.get(), 1);
                ClusterWritable clusterWritable = new ClusterWritable();
                clusterWritable.setValue(newCluster);
                chosenClusters.put(index, clusterWritable);
              }
            }
          }
        }
      }

      try {
        for (Integer key : maxEigens.keySet()) {
          writer.append(chosenTexts.get(key), chosenClusters.get(key));
        }
        log.info("EigenSeedGenerator:: Wrote {} Klusters to {}", chosenTexts.size(), outFile);
      } finally {
        Closeables.close(writer, false);
      }
    }

    return outFile;
  }

}
