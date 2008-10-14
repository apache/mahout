package org.apache.mahout.classifier.bayes.common;

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

import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;

public class BayesTfIdfMapper extends MapReduceBase implements
    Mapper<Text, FloatWritable, Text, FloatWritable> {

  private static final Logger log = LoggerFactory.getLogger(BayesTfIdfMapper.class);  

  public HashMap<String, Float> labelDocumentCounts = null;
  String labelDocumentCountString =" ";

  /**
   * We need to calculate the Tf-Idf of each feature in each label
   * 
   * @param key The label,feature pair (can either be the freq Count or the term
   *        Document count
   * @param value
   * @param output
   * @param reporter
   * @throws IOException
   */
  public void map(Text key, FloatWritable value,
      OutputCollector<Text, FloatWritable> output, Reporter reporter)
      throws IOException {
 
    String labelFeaturePair = key.toString();

    if (labelFeaturePair.startsWith("-")) { // if it is the termDocumentCount
      labelFeaturePair = labelFeaturePair.substring(1);
      String label = labelFeaturePair.split(",")[0];
      
      if(labelDocumentCounts.containsKey(label)==false){
        
        throw new IOException(label);
      }
      
      float labelDocumentCount = labelDocumentCounts.get(label);
      float logIdf = (float) Math.log(labelDocumentCount  / value.get());
      
      output.collect(new Text(labelFeaturePair), new FloatWritable(logIdf));
    } else if (labelFeaturePair.startsWith(",")) {
      output.collect(new Text("*vocabCount"), new FloatWritable(1.0f));
    } else {
      output.collect(key, value);
    }
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      if (labelDocumentCounts == null){
        labelDocumentCounts = new HashMap<String, Float>();

        DefaultStringifier<HashMap<String,Float>> mapStringifier =
            new DefaultStringifier<HashMap<String,Float>>(job,GenericsUtil.getClass(labelDocumentCounts));

        labelDocumentCountString = mapStringifier.toString(labelDocumentCounts);  
        labelDocumentCountString = job.get("cnaivebayes.labelDocumentCounts", labelDocumentCountString);
        
        labelDocumentCounts = mapStringifier.fromString(labelDocumentCountString);
      }
    } catch(IOException ex){
      log.warn(ex.toString(), ex);
    }
  }
}
