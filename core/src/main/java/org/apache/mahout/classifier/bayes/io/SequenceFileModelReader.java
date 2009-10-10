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

package org.apache.mahout.classifier.bayes.io;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.classifier.bayes.datastore.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * This Class reads the different interim files created during the Training
 * stage as well as the Model File during testing.
 */
public class SequenceFileModelReader {

  private static final Logger log = LoggerFactory
      .getLogger(SequenceFileModelReader.class);

  private SequenceFileModelReader() {
  }

  public static void loadModel(InMemoryBayesDatastore datastore, FileSystem fs,
      Parameters params, Configuration conf) throws IOException {

    loadFeatureWeights(datastore, fs, new Path(params.get("sigma_j")), conf);
    loadLabelWeights(datastore, fs, new Path(params.get("sigma_k")), conf);
    loadSumWeight(datastore, fs, new Path(params.get("sigma_kSigma_j")), conf);
    loadThetaNormalizer(datastore, fs, new Path(params.get("thetaNormalizer")),
        conf);
    loadWeightMatrix(datastore, fs, new Path(params.get("weight")), conf);

  }

  public static void loadWeightMatrix(InMemoryBayesDatastore datastore,
      FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is label,feature
      while (reader.next(key, value)) {

        datastore.loadFeatureWeight(key.stringAt(2), key.stringAt(1), value
            .get());

      }
    }
  }

  public static void loadFeatureWeights(InMemoryBayesDatastore datastore,
      FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      long count = 0;
      while (reader.next(key, value)) {

        if (key.stringAt(0).equals(BayesConstants.FEATURE_SUM)) { // Sum of
                                                                  // weights for
                                                                  // a Feature
          datastore.setSumFeatureWeight(key.stringAt(1), value.get());
          count++;
          if (count % 50000 == 0) {
            log.info("Read {} feature weights", count);
          }
        }
      }
    }
  }

  public static void loadLabelWeights(InMemoryBayesDatastore datastore,
      FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      long count = 0;
      while (reader.next(key, value)) {
        if (key.stringAt(0).equals(BayesConstants.LABEL_SUM)) { // Sum of
                                                                // weights in a
                                                                // Label
          datastore.setSumLabelWeight(key.stringAt(1), value.get());
          count++;
          if (count % 10000 == 0) {
            log.info("Read {} label weights", count);
          }
        }
      }
    }
  }

  public static void loadThetaNormalizer(InMemoryBayesDatastore datastore,
      FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      long count = 0;
      while (reader.next(key, value)) {
        if (key.stringAt(0).equals(BayesConstants.LABEL_THETA_NORMALIZER)) { // Sum
                                                                             // of
                                                                             // weights
                                                                             // in
                                                                             // a
                                                                             // Label
          datastore.setThetaNormalizer(key.stringAt(1), value.get());
          count++;
          if (count % 50000 == 0) {
            log.info("Read {} theta norms", count);
          }
        }
      }
    }
  }

  public static void loadSumWeight(InMemoryBayesDatastore datastore,
      FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is _label
      while (reader.next(key, value)) {

        if (key.stringAt(0).equals(BayesConstants.TOTAL_SUM)) { // Sum of
                                                                // weights for
          // all Features and all Labels
          datastore.setSigma_jSigma_k(value.get());
          log.info("{}", value.get());
        }
      }
    }
  }

  public static Map<String, Double> readLabelSums(FileSystem fs,
      Path pathPattern, Configuration conf) throws IOException {
    Map<String, Double> labelSum = new HashMap<String, Double>();
    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);

    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        if (key.stringAt(0).equals(BayesConstants.LABEL_SUM)) { // Sum of counts
                                                                // of labels
          labelSum.put(key.stringAt(1), value.get());
        }

      }
    }

    return labelSum;
  }

  public static Map<String, Double> readLabelDocumentCounts(FileSystem fs,
      Path pathPattern, Configuration conf) throws IOException {
    Map<String, Double> labelDocumentCounts = new HashMap<String, Double>();
    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        if (key.stringAt(0).equals(BayesConstants.LABEL_COUNT)) { // Count of
                                                                  // Documents
                                                                  // in a Label
          labelDocumentCounts.put(key.stringAt(1), value.get());
        }

      }
    }

    return labelDocumentCounts;
  }

  public static double readSigma_jSigma_k(FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Map<String, Double> weightSum = new HashMap<String, Double>();
    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is *
      while (reader.next(key, value)) {
        if (weightSum.size() > 1) {
          throw new IOException("Incorrect Sum File");
        } else if (key.stringAt(0).equals(BayesConstants.TOTAL_SUM)) {
          weightSum.put(BayesConstants.TOTAL_SUM, value.get());
        }

      }
    }

    return weightSum.get(BayesConstants.TOTAL_SUM);
  }

  public static double readVocabCount(FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Map<String, Double> weightSum = new HashMap<String, Double>();
    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is *
      while (reader.next(key, value)) {
        if (weightSum.size() > 1) {
          throw new IOException("Incorrect vocabCount File");
        }
        if (key.stringAt(0).equals(BayesConstants.FEATURE_SET_SIZE)) {
          weightSum.put(BayesConstants.FEATURE_SET_SIZE, value.get());
        }

      }
    }

    return weightSum.get(BayesConstants.FEATURE_SET_SIZE);
  }
}
