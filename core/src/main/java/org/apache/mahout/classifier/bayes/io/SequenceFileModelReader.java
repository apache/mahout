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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.bayes.datastore.InMemoryBayesDatastore;
import org.apache.mahout.common.Parameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * This Class reads the different interim  files created during the Training stage as well as the Model File during
 * testing.
 */
public class SequenceFileModelReader {

  private static final Logger log = LoggerFactory.getLogger(SequenceFileModelReader.class);

  private SequenceFileModelReader() {
  }

  public static void loadModel(InMemoryBayesDatastore datastore, FileSystem fs, Parameters params,
                               Configuration conf) throws IOException {

    loadFeatureWeights(datastore, fs, new Path(params.get("sigma_j")), conf);
    loadLabelWeights(datastore, fs, new Path(params.get("sigma_k")), conf); 
    loadSumWeight(datastore, fs, new Path(params.get("sigma_kSigma_j")), conf); 
    loadThetaNormalizer(datastore, fs, new Path(params.get("thetaNormalizer")), conf); 
    loadWeightMatrix(datastore, fs, new Path(params.get("weight")), conf);


  }

  public static void loadWeightMatrix(InMemoryBayesDatastore datastore, FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        int idx = keyStr.indexOf(',');
        if (idx != -1) {
          datastore.loadFeatureWeight(keyStr.substring(idx + 1),keyStr.substring(0, idx), value.get());
        }

      }
    }
  }

  public static void loadFeatureWeights(InMemoryBayesDatastore datastore, FileSystem fs, Path pathPattern,
                                        Configuration conf) throws IOException {

    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      long count = 0;
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.charAt(0) == ',') { // Sum of weights for a Feature
          datastore.setSumFeatureWeight(keyStr.substring(1),
              value.get());
          count++;
          if (count % 50000 == 0){
            log.info("Read {} feature weights", count);
          }
        }
      }
    }
  }

  public static void loadLabelWeights(InMemoryBayesDatastore datastore,FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {

    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      long count = 0;
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.charAt(0) == '_') { // Sum of weights in a Label
          datastore.setSumLabelWeight(keyStr.substring(1), value
              .get());
          count++;
          if (count % 10000 == 0){
            log.info("Read {} label weights", count);
          }
        }
      }
    }
  }

  
  public static void loadThetaNormalizer(InMemoryBayesDatastore datastore,FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {

    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      long count = 0;
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (keyStr.charAt(0) == '_') { // Sum of weights in a Label
          datastore.setThetaNormalizer(keyStr.substring(1), value
              .get());
          count++;
          if (count % 50000 == 0){
            log.info("Read {} theta norms", count);
          }
        }
      }
    }
  }

  public static void loadSumWeight(InMemoryBayesDatastore datastore, FileSystem fs, Path pathPattern,
                                   Configuration conf) throws IOException {

    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.charAt(0) == '*') { // Sum of weights for all Feature
          // and all Labels
          datastore.setSigma_jSigma_k(value.get());
          log.info("{}", value.get());
        }
      }
    }
  }

  public static Map<String, Double> readLabelSums(FileSystem fs, Path pathPattern, Configuration conf) throws IOException {
    Map<String, Double> labelSum = new HashMap<String, Double>();
    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);

    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (keyStr.charAt(0) == '_') { // Sum of weights of labels
          labelSum.put(keyStr.substring(1), value.get());
        }

      }
    }

    return labelSum;
  }

  public static Map<String, Double> readLabelDocumentCounts(FileSystem fs, Path pathPattern, Configuration conf)
      throws IOException {
    Map<String, Double> labelDocumentCounts = new HashMap<String, Double>();
    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (keyStr.charAt(0) == '_') { // Count of Documents in a Label
          labelDocumentCounts.put(keyStr.substring(1), value.get());
        }

      }
    }

    return labelDocumentCounts;
  }

  public static double readSigma_jSigma_k(FileSystem fs, Path pathPattern,
                                          Configuration conf) throws IOException {
    Map<String, Double> weightSum = new HashMap<String, Double>();
    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is *
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (weightSum.size() > 1) {
          throw new IOException("Incorrect Sum File");
        } else if (keyStr.charAt(0) == '*') {
          weightSum.put(keyStr, value.get());
        }

      }
    }

    return weightSum.get("*");
  }

  public static double readVocabCount(FileSystem fs, Path pathPattern,
                                      Configuration conf) throws IOException {
    Map<String, Double> weightSum = new HashMap<String, Double>();
    Writable key = new Text();
    DoubleWritable value = new DoubleWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is *
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (weightSum.size() > 1) {
          throw new IOException("Incorrect vocabCount File");
        }
        if (keyStr.charAt(0) == '*') {
          weightSum.put(keyStr, value.get());
        }

      }
    }

    return weightSum.get("*vocabCount");
  }

}
