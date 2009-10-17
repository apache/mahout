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

package org.apache.mahout.classifier.bayes.mapreduce.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.KeyValueTextInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.classifier.bayes.common.BayesParameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Create and run the Bayes Classifier */
public class BayesClassifierDriver {

  private static final Logger log = LoggerFactory.getLogger(BayesClassifierDriver.class);
  private BayesClassifierDriver() {
  }

  /**
   * Run the job
   * 
   * @param params The Job parameters containing the gramSize, input output folders, defaultCat, encoding
   */
  public static final void runJob(BayesParameters params) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(BayesClassifierDriver.class);
    conf.setJobName("Bayes Classifier Driver running over input: " + params.get("testDirPath"));
    conf.setOutputKeyClass(StringTuple.class);
    conf.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.setInputPaths(conf, new Path(params.get("testDirPath")));
    Path outPath = new Path(params.get("testDirPath") + "-output");
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setInputFormat(KeyValueTextInputFormat.class);
    conf.setMapperClass(BayesClassifierMapper.class);
    conf.setCombinerClass(BayesClassifierReducer.class);
    conf.setReducerClass(BayesClassifierReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    
    conf.set("io.serializations",
        "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    conf.set("bayes.parameters", params.toString());

    client.setConf(conf);
    JobClient.runJob(conf);

    Path outputFiles = new Path(outPath.toString() + "/part*");
    ConfusionMatrix matrix = readResult(dfs, outputFiles, conf, params);
    log.info("{}",matrix.summarize());
  }

  private static ConfusionMatrix readResult(FileSystem fs, Path pathPattern, Configuration conf, BayesParameters params)
      throws IOException {
   
    StringTuple key = new StringTuple();
    DoubleWritable value = new DoubleWritable();
    String defaultLabel = params.get("defaultCat");
    FileStatus[] outputFiles = fs.globStatus(pathPattern);
    Map<String, Map<String, Integer>> confusionMatrix = new HashMap<String, Map<String, Integer>>();
    
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      while (reader.next(key, value)) {        
        String correctLabel = key.stringAt(1);
        String classifiedLabel = key.stringAt(2);
        Map<String, Integer> rowMatrix = confusionMatrix.get(correctLabel);
        if(rowMatrix == null)
          rowMatrix = new HashMap<String, Integer>();        
        Integer count = Double.valueOf(value.get()).intValue();
        rowMatrix.put(classifiedLabel, count);
        confusionMatrix.put(correctLabel, rowMatrix);
        
      }
    }
    
    ConfusionMatrix matrix = new ConfusionMatrix(confusionMatrix.keySet(), defaultLabel);
    for(Map.Entry<String, Map<String, Integer>> correctLabelSet:confusionMatrix.entrySet())
    {
      Map<String, Integer> rowMatrix = correctLabelSet.getValue();
      for(Map.Entry<String, Integer> classifiedLabelSet : rowMatrix.entrySet())
      {
        matrix.addInstance(correctLabelSet.getKey(), classifiedLabelSet.getKey());
        matrix.putCount(correctLabelSet.getKey(), classifiedLabelSet.getKey(), classifiedLabelSet.getValue());
      }
    }
    return matrix;
    
  }
}
