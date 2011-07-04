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

package org.apache.mahout.classifier.naivebayes.training;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.IOException;
import java.net.URI;
import java.util.Map;

public class TrainUtils {

  private TrainUtils() {}

  static NaiveBayesModel readModelFromTempDir(Path base, Configuration conf) {

    Vector scoresPerLabel = null;
    Vector perlabelThetaNormalizer = null;
    Vector scoresPerFeature = null;
    Matrix scoresPerLabelAndFeature;
    float alphaI;

    alphaI = conf.getFloat(ThetaMapper.ALPHA_I, 1.0f);

    // read feature sums and label sums
    for (Pair<Text,VectorWritable> record : new SequenceFileDirIterable<Text, VectorWritable>(
        new Path(base, TrainNaiveBayesJob.WEIGHTS), PathType.LIST, PathFilters.partFilter(), conf)) {
      String key = record.getFirst().toString();
      VectorWritable value = record.getSecond();
      if (key.equals(TrainNaiveBayesJob.WEIGHTS_PER_FEATURE)) {
        scoresPerFeature = value.get();
      } else if (key.equals(TrainNaiveBayesJob.WEIGHTS_PER_LABEL)) {
        scoresPerLabel = value.get();
      }
    }

    Preconditions.checkNotNull(scoresPerFeature);
    Preconditions.checkNotNull(scoresPerLabel);

    scoresPerLabelAndFeature = new SparseMatrix(new int[] { scoresPerLabel.size(), scoresPerFeature.size() });
    for (Pair<IntWritable,VectorWritable> entry : new SequenceFileDirIterable<IntWritable,VectorWritable>(
        new Path(base, TrainNaiveBayesJob.SUMMED_OBSERVATIONS), PathType.LIST, PathFilters.partFilter(), conf)) {
      scoresPerLabelAndFeature.assignRow(entry.getFirst().get(), entry.getSecond().get());
    }

    for (Pair<Text,VectorWritable> entry : new SequenceFileDirIterable<Text,VectorWritable>(
        new Path(base, TrainNaiveBayesJob.THETAS), PathType.LIST, PathFilters.partFilter(), conf)) {
      if (entry.getFirst().toString().equals(TrainNaiveBayesJob.LABEL_THETA_NORMALIZER)) {
        perlabelThetaNormalizer = entry.getSecond().get();
      }
    }

    Preconditions.checkNotNull(perlabelThetaNormalizer);

    return new NaiveBayesModel(scoresPerLabelAndFeature, scoresPerFeature, scoresPerLabel, perlabelThetaNormalizer,
        alphaI);
  }

  protected static void setSerializations(Configuration conf) {
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
  }

  protected static void cacheFiles(Path fileToCache, Configuration conf) {
    DistributedCache.setCacheFiles(new URI[] { fileToCache.toUri() }, conf);
  }

  /** Write the list of labels into a map file */
  protected static void writeLabelIndex(Configuration conf, Iterable<String> labels, Path indexPath)
      throws IOException {
    FileSystem fs = FileSystem.get(indexPath.toUri(), conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, indexPath, Text.class, IntWritable.class);
    try {
      int i = 0;
      for (String label : labels) {
        writer.append(new Text(label), new IntWritable(i++));
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  private static Path cachedFile(Configuration conf) throws IOException {
    return new Path(DistributedCache.getCacheFiles(conf)[0].getPath());
  }

  protected static OpenObjectIntHashMap<String> readIndexFromCache(Configuration conf) throws IOException {
    OpenObjectIntHashMap<String> index = new OpenObjectIntHashMap<String>();
    for (Pair<Writable,IntWritable> entry : new SequenceFileIterable<Writable,IntWritable>(cachedFile(conf), conf)) {
      index.put(entry.getFirst().toString(), entry.getSecond().get());
    }
    return index;
  }

  protected static Map<String,Vector> readScoresFromCache(Configuration conf) throws IOException {
    Map<String,Vector> sumVectors = Maps.newHashMap();
    for (Pair<Text,VectorWritable> entry : new SequenceFileDirIterable<Text,VectorWritable>(cachedFile(conf),
        PathType.LIST, PathFilters.partFilter(), conf)) {
      sumVectors.put(entry.getFirst().toString(), entry.getSecond().get());
    }
    return sumVectors;
  }
}
