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

package org.apache.mahout.classifier.naivebayes;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.naivebayes.training.ThetaMapper;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.common.HadoopUtil;
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

public final class BayesUtils {

  private static final Pattern SLASH = Pattern.compile("/");

  private BayesUtils() {}

  public static NaiveBayesModel readModelFromDir(Path base, Configuration conf) {

    float alphaI = conf.getFloat(ThetaMapper.ALPHA_I, 1.0f);
    boolean isComplementary = conf.getBoolean(NaiveBayesModel.COMPLEMENTARY_MODEL, true);

    // read feature sums and label sums
    Vector scoresPerLabel = null;
    Vector scoresPerFeature = null;
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

    Matrix scoresPerLabelAndFeature = new SparseMatrix(scoresPerLabel.size(), scoresPerFeature.size());
    for (Pair<IntWritable,VectorWritable> entry : new SequenceFileDirIterable<IntWritable,VectorWritable>(
        new Path(base, TrainNaiveBayesJob.SUMMED_OBSERVATIONS), PathType.LIST, PathFilters.partFilter(), conf)) {
      scoresPerLabelAndFeature.assignRow(entry.getFirst().get(), entry.getSecond().get());
    }
    
    // perLabelThetaNormalizer is only used by the complementary model, we do not instantiate it for the standard model
    Vector perLabelThetaNormalizer = null;
    if (isComplementary) {
      perLabelThetaNormalizer=scoresPerLabel.like();    
      for (Pair<Text,VectorWritable> entry : new SequenceFileDirIterable<Text,VectorWritable>(
          new Path(base, TrainNaiveBayesJob.THETAS), PathType.LIST, PathFilters.partFilter(), conf)) {
        if (entry.getFirst().toString().equals(TrainNaiveBayesJob.LABEL_THETA_NORMALIZER)) {
          perLabelThetaNormalizer = entry.getSecond().get();
        }
      }
      Preconditions.checkNotNull(perLabelThetaNormalizer);
    }
     
    return new NaiveBayesModel(scoresPerLabelAndFeature, scoresPerFeature, scoresPerLabel, perLabelThetaNormalizer,
        alphaI, isComplementary);
  }

  /** Write the list of labels into a map file */
  public static int writeLabelIndex(Configuration conf, Iterable<String> labels, Path indexPath)
    throws IOException {
    FileSystem fs = FileSystem.get(indexPath.toUri(), conf);
    int i = 0;
    try (SequenceFile.Writer writer =
           SequenceFile.createWriter(fs.getConf(), SequenceFile.Writer.file(indexPath),
             SequenceFile.Writer.keyClass(Text.class), SequenceFile.Writer.valueClass(IntWritable.class))) {
      for (String label : labels) {
        writer.append(new Text(label), new IntWritable(i++));
      }
    }
    return i;
  }

  public static int writeLabelIndex(Configuration conf, Path indexPath,
                                    Iterable<Pair<Text,IntWritable>> labels) throws IOException {
    FileSystem fs = FileSystem.get(indexPath.toUri(), conf);
    Collection<String> seen = new HashSet<>();
    int i = 0;
    try (SequenceFile.Writer writer =
           SequenceFile.createWriter(fs.getConf(), SequenceFile.Writer.file(indexPath),
             SequenceFile.Writer.keyClass(Text.class), SequenceFile.Writer.valueClass(IntWritable.class))){
      for (Object label : labels) {
        String theLabel = SLASH.split(((Pair<?, ?>) label).getFirst().toString())[1];
        if (!seen.contains(theLabel)) {
          writer.append(new Text(theLabel), new IntWritable(i++));
          seen.add(theLabel);
        }
      }
    }
    return i;
  }

  public static Map<Integer, String> readLabelIndex(Configuration conf, Path indexPath) {
    Map<Integer, String> labelMap = new HashMap<>();
    for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(indexPath, true, conf)) {
      labelMap.put(pair.getSecond().get(), pair.getFirst().toString());
    }
    return labelMap;
  }

  public static OpenObjectIntHashMap<String> readIndexFromCache(Configuration conf) throws IOException {
    OpenObjectIntHashMap<String> index = new OpenObjectIntHashMap<>();
    for (Pair<Writable,IntWritable> entry
        : new SequenceFileIterable<Writable,IntWritable>(HadoopUtil.getSingleCachedFile(conf), conf)) {
      index.put(entry.getFirst().toString(), entry.getSecond().get());
    }
    return index;
  }

  public static Map<String,Vector> readScoresFromCache(Configuration conf) throws IOException {
    Map<String,Vector> sumVectors = new HashMap<>();
    for (Pair<Text,VectorWritable> entry
        : new SequenceFileDirIterable<Text,VectorWritable>(HadoopUtil.getSingleCachedFile(conf),
          PathType.LIST, PathFilters.partFilter(), conf)) {
      sumVectors.put(entry.getFirst().toString(), entry.getSecond().get());
    }
    return sumVectors;
  }


}
