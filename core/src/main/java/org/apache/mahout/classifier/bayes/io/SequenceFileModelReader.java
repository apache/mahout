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
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * This Class reads the different interim  files created during the Training stage
 * as well as the Model File during testing.
 */
public class SequenceFileModelReader {

  private static final Logger log = LoggerFactory.getLogger(SequenceFileModelReader.class);  

  public Model loadModel(Model model, FileSystem fs, Map<String, Path> pathPatterns,
      Configuration conf) throws IOException {

    loadFeatureWeights(model, fs, pathPatterns.get("sigma_j"), conf);
    loadLabelWeights(model, fs, pathPatterns.get("sigma_k"), conf); 
    loadSumWeight(model, fs, pathPatterns.get("sigma_kSigma_j"), conf); 
    loadThetaNormalizer(model, fs, pathPatterns.get("thetaNormalizer"), conf); 
    
   
    model.initializeWeightMatrix();
    
    loadWeightMatrix(model, fs, pathPatterns.get("weight"), conf);
    model.InitializeNormalizer();
    //model.GenerateComplementaryModel();
    return model;
  }

  public void loadWeightMatrix(Model model, FileSystem fs, Path pathPattern, Configuration conf) throws IOException {

    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        int idx = keyStr.indexOf(",");
        if (idx != -1) {
          model.loadFeatureWeight(keyStr.substring(0, idx), keyStr.substring(idx + 1), value.get());
        }

      }
    }
  }

  public Model loadFeatureWeights(Model model, FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {

    Writable key = new Text();
    FloatWritable value = new FloatWritable();
    
    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.startsWith(",")) { // Sum of weights for a Feature
          model.setSumFeatureWeight(keyStr.substring(1),
              value.get());
        }
      }
    }
    return model;
  }

  public Model loadLabelWeights(Model model,FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.startsWith("_")) { // Sum of weights in a Label
          model.setSumLabelWeight(keyStr.substring(1), value
              .get());
        }
      }
    }

    return model;
  }
  
  public Model loadThetaNormalizer(Model model,FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();        
        if (keyStr.startsWith("_")) { // Sum of weights in a Label
          model.setThetaNormalizer(keyStr.substring(1), value
              .get());
        }
      }
    }

    return model;
  }

  public Model loadSumWeight(Model model, FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {

    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();

        if (keyStr.startsWith("*")) { // Sum of weights for all Feature
          // and all Labels
          model.setSigma_jSigma_k(value.get());
          log.info("{}", value.get());
        }
      }
    }
    return model;
  }

  public void createMapFile(FileSystem fs, Path pathPattern, Configuration conf)
      throws IOException {

    Writable key = new Text();
    FloatWritable value = new FloatWritable();
    MapFile.Writer writer = new MapFile.Writer(conf, fs, "data.mapfile",
        Text.class, FloatWritable.class);
    MapFile.Writer.setIndexInterval(conf, 3);

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      log.info("{}", path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        // TODO srowen says we should probably collapse these empty branches?
        if (keyStr.startsWith("_")) {

        } else if (keyStr.startsWith(",")) {

        } else if (keyStr.startsWith("*")) {

        } else {
          int idx = keyStr.indexOf(",");
          if (idx != -1) {
            // TODO srowen says data is not used?
            Map<String, Float> data = new HashMap<String, Float>();
            data.put(keyStr.substring(0, idx), value.get());
            writer.append(new Text(key.toString()), value);
          }
        }
      }
    }
    writer.close();
    // return model;
  }

  public Map<String, Float> readLabelSums(FileSystem fs, Path pathPattern, Configuration conf) throws IOException {
    Map<String, Float> labelSum = new HashMap<String, Float>();
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
   
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (keyStr.startsWith("_")) { // Sum of weights of labels
          labelSum.put(keyStr.substring(1), value.get());
        }

      }
    }

    return labelSum;
  }

  public Map<String, Float> readLabelDocumentCounts(FileSystem fs, Path pathPattern, Configuration conf)
      throws IOException {
    Map<String, Float> labelDocumentCounts = new HashMap<String, Float>();
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is either _label_ or label,feature
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (keyStr.startsWith("_")) { // Count of Documents in a Label
          labelDocumentCounts.put(keyStr.substring(1), value.get());
        }

      }
    }

    return labelDocumentCounts;
  }

  public Float readSigma_jSigma_k(FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Map<String, Float> weightSum = new HashMap<String, Float>();
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // the key is *
      while (reader.next(key, value)) {
        String keyStr = key.toString();
        if (weightSum.size() > 1) {
          throw new IOException("Incorrect Sum File");
        } else if (keyStr.startsWith("*")) {
          weightSum.put(keyStr, value.get());
        }

      }
    }

    return weightSum.get("*");
  }

  public Float readVocabCount(FileSystem fs, Path pathPattern,
      Configuration conf) throws IOException {
    Map<String, Float> weightSum = new HashMap<String, Float>();
    Writable key = new Text();
    FloatWritable value = new FloatWritable();

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
        if (keyStr.startsWith("*")) {
          weightSum.put(keyStr, value.get());
        }

      }
    }

    return weightSum.get("*vocabCount");
  }

}
