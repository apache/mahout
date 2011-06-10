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

package org.apache.mahout.classifier.bayes.mapreduce.common;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.SequenceFileModelReader;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** The Driver which drives the Tf-Idf Generation */
public class BayesTfIdfDriver implements BayesJob {
  
  private static final Logger log = LoggerFactory.getLogger(BayesTfIdfDriver.class);

  @Override
  public void runJob(Path input, Path output, BayesParameters params) throws IOException {
    
    Configurable client = new JobClient();
    JobConf conf = new JobConf(BayesWeightSummerDriver.class);
    conf.setJobName("TfIdf Driver running over input: " + input);
    
    conf.setOutputKeyClass(StringTuple.class);
    conf.setOutputValueClass(DoubleWritable.class);
    
    FileInputFormat.addInputPath(conf, new Path(output, "trainer-termDocCount"));
    FileInputFormat.addInputPath(conf, new Path(output, "trainer-wordFreq"));
    FileInputFormat.addInputPath(conf, new Path(output, "trainer-featureCount"));
    Path outPath = new Path(output, "trainer-tfIdf");
    FileOutputFormat.setOutputPath(conf, outPath);
    
    // conf.setNumMapTasks(100);
    
    conf.setJarByClass(BayesTfIdfDriver.class);
    
    conf.setMapperClass(BayesTfIdfMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(BayesTfIdfReducer.class);
    
    conf.setReducerClass(BayesTfIdfReducer.class);
    
    conf.setOutputFormat(BayesTfIdfOutputFormat.class);
    
    conf.set("io.serializations",
             "org.apache.hadoop.io.serializer.JavaSerialization,"
                 + "org.apache.hadoop.io.serializer.WritableSerialization");
    // Dont ever forget this. People should keep track of how hadoop conf
    // parameters and make or break a piece of code
    HadoopUtil.delete(conf, outPath);
    Path interimFile = new Path(output, "trainer-docCount/part-*");
    
    Map<String,Double> labelDocumentCounts = SequenceFileModelReader.readLabelDocumentCounts(interimFile, conf);
    
    DefaultStringifier<Map<String,Double>> mapStringifier = new DefaultStringifier<Map<String,Double>>(conf,
        GenericsUtil.getClass(labelDocumentCounts));
    
    String labelDocumentCountString = mapStringifier.toString(labelDocumentCounts);
    log.info("Counts of documents in Each Label");
    Map<String,Double> c = mapStringifier.fromString(labelDocumentCountString);
    log.info("{}", c);
    
    conf.set("cnaivebayes.labelDocumentCounts", labelDocumentCountString);
    log.info(params.print());
    conf.set("bayes.parameters", params.toString());
    
    client.setConf(conf);
    
    JobClient.runJob(conf);
  }
}
