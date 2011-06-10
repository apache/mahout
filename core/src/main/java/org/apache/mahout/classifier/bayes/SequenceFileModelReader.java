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

package org.apache.mahout.classifier.bayes;

import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This Class reads the different interim files created during the Training stage as well as the Model File
 * during testing.
 */
public final class SequenceFileModelReader {
  
  private static final Logger log = LoggerFactory.getLogger(SequenceFileModelReader.class);
  
  private SequenceFileModelReader() { }
  
  public static void loadModel(InMemoryBayesDatastore datastore, Parameters params, Configuration conf) {
    loadFeatureWeights(datastore, new Path(params.get("sigma_j")), conf);
    loadLabelWeights(datastore, new Path(params.get("sigma_k")), conf);
    loadSumWeight(datastore, new Path(params.get("sigma_kSigma_j")), conf);
    loadThetaNormalizer(datastore, new Path(params.get("thetaNormalizer")), conf);
    loadWeightMatrix(datastore, new Path(params.get("weight")), conf);
  }
  
  public static void loadWeightMatrix(InMemoryBayesDatastore datastore, Path pathPattern, Configuration conf) {
    // the key is label,feature
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      datastore.loadFeatureWeight(key.stringAt(2), key.stringAt(1), value.get());
    }
  }
  
  public static void loadFeatureWeights(InMemoryBayesDatastore datastore, Path pathPattern, Configuration conf) {
    // the key is either _label_ or label,feature
    long count = 0;
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      // Sum of weights for a Feature
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (key.stringAt(0).equals(BayesConstants.FEATURE_SUM)) {
        datastore.setSumFeatureWeight(key.stringAt(1), value.get());
        if (++count % 50000 == 0) {
          log.info("Read {} feature weights", count);
        }
      }
    }
  }
  
  public static void loadLabelWeights(InMemoryBayesDatastore datastore, Path pathPattern, Configuration conf) {
    long count = 0;
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      // Sum of weights in a Label
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (key.stringAt(0).equals(BayesConstants.LABEL_SUM)) {
        datastore.setSumLabelWeight(key.stringAt(1), value.get());
        if (++count % 10000 == 0) {
          log.info("Read {} label weights", count);
        }
      }
    }
  }
  
  public static void loadThetaNormalizer(InMemoryBayesDatastore datastore, Path pathPattern, Configuration conf) {
    long count = 0;
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      // Sum of weights in a Label
      if (key.stringAt(0).equals(BayesConstants.LABEL_THETA_NORMALIZER)) {
        datastore.setThetaNormalizer(key.stringAt(1), value.get());
        if (++count % 50000 == 0) {
          log.info("Read {} theta norms", count);
        }
      }
    }
  }
  
  public static void loadSumWeight(InMemoryBayesDatastore datastore, Path pathPattern, Configuration conf) {
    // the key is _label
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (key.stringAt(0).equals(BayesConstants.TOTAL_SUM)) {
        // Sum of weights for all Features and all Labels
        datastore.setSigmaJSigmaK(value.get());
        log.info("{}", value.get());
      }
    }
  }
  
  public static Map<String,Double> readLabelSums(Path pathPattern, Configuration conf) {
    Map<String,Double> labelSum = Maps.newHashMap();
    // the key is either _label_ or label,feature
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (key.stringAt(0).equals(BayesConstants.LABEL_SUM)) {
        // Sum of counts of labels
        labelSum.put(key.stringAt(1), value.get());
      }
    }
    return labelSum;
  }
  
  public static Map<String,Double> readLabelDocumentCounts(Path pathPattern, Configuration conf) {
    Map<String,Double> labelDocumentCounts = Maps.newHashMap();
    // the key is either _label_ or label,feature
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      // Count of Documents in a Label
      if (key.stringAt(0).equals(BayesConstants.LABEL_COUNT)) {
        labelDocumentCounts.put(key.stringAt(1), value.get());
      }
    }
    return labelDocumentCounts;
  }
  
  public static double readSigmaJSigmaK(Path pathPattern, Configuration conf) {
    Map<String,Double> weightSum = Maps.newHashMap();
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB,
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (weightSum.size() > 1) {
        throw new IllegalStateException("Incorrect Sum File");
      } else if (key.stringAt(0).equals(BayesConstants.TOTAL_SUM)) {
        weightSum.put(BayesConstants.TOTAL_SUM, value.get());
      }
    }
    return weightSum.get(BayesConstants.TOTAL_SUM);
  }
  
  public static double readVocabCount(Path pathPattern, Configuration conf) {
    Map<String,Double> weightSum = Maps.newHashMap();
    for (Pair<StringTuple,DoubleWritable> record
         : new SequenceFileDirIterable<StringTuple,DoubleWritable>(pathPattern,
                                                                   PathType.GLOB, 
                                                                   null,
                                                                   null,
                                                                   true,
                                                                   conf)) {
      if (weightSum.size() > 1) {
        throw new IllegalStateException("Incorrect vocabCount File");
      }
      StringTuple key = record.getFirst();
      DoubleWritable value = record.getSecond();
      if (key.stringAt(0).equals(BayesConstants.FEATURE_SET_SIZE)) {
        weightSum.put(BayesConstants.FEATURE_SET_SIZE, value.get());
      }
    }
    return weightSum.get(BayesConstants.FEATURE_SET_SIZE);
  }
}
