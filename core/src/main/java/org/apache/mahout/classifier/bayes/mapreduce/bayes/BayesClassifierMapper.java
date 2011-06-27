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

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.CBayesAlgorithm;
import org.apache.mahout.classifier.bayes.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.Algorithm;
import org.apache.mahout.classifier.bayes.Datastore;
import org.apache.mahout.classifier.bayes.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.classifier.bayes.ClassifierContext;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.nlp.NGrams;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads the input train set(preprocessed using the {@link org.apache.mahout.classifier.BayesFileFormatter}).
 */
public class BayesClassifierMapper extends MapReduceBase implements
    Mapper<Text,Text,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesClassifierMapper.class);
  private static final DoubleWritable ONE = new DoubleWritable(1.0);
  
  private int gramSize = 1;
  
  private ClassifierContext classifier;
  
  private String defaultCategory;
  
  /**
   * Parallel Classification
   * 
   * @param key
   *          The label
   * @param value
   *          the features (all unique) associated w/ this label
   * @param output
   *          The OutputCollector to write the results to
   * @param reporter
   *          Reports status back to hadoop
   */
  @Override
  public void map(Text key, Text value,
                  OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    List<String> ngrams = new NGrams(value.toString(), gramSize).generateNGramsWithoutLabel();
    
    try {
      ClassifierResult result = classifier.classifyDocument(ngrams.toArray(new String[ngrams.size()]),
        defaultCategory);
      
      String correctLabel = key.toString();
      String classifiedLabel = result.getLabel();
      
      StringTuple outputTuple = new StringTuple(BayesConstants.CLASSIFIER_TUPLE);
      outputTuple.add(correctLabel);
      outputTuple.add(classifiedLabel);
      
      output.collect(outputTuple, ONE);
    } catch (InvalidDatastoreException e) {
      throw new IOException(e);
    }
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      BayesParameters params = new BayesParameters(job.get("bayes.parameters", ""));
      log.info("Bayes Parameter {}", params.print());
      log.info("{}", params.print());
      Algorithm algorithm;
      Datastore datastore;
      
      if ("hdfs".equals(params.get("dataSource"))) {
        if ("bayes".equalsIgnoreCase(params.get("classifierType"))) {
          log.info("Testing Bayes Classifier");
          algorithm = new BayesAlgorithm();
          datastore = new InMemoryBayesDatastore(params);
        } else if ("cbayes".equalsIgnoreCase(params.get("classifierType"))) {
          log.info("Testing Complementary Bayes Classifier");
          algorithm = new CBayesAlgorithm();
          datastore = new InMemoryBayesDatastore(params);
        } else {
          throw new IllegalArgumentException("Unrecognized classifier type: " + params.get("classifierType"));
        }
        
      } else {
        throw new IllegalArgumentException("Unrecognized dataSource type: " + params.get("dataSource"));
      }
      classifier = new ClassifierContext(algorithm, datastore);
      classifier.initialize();
      
      defaultCategory = params.get("defaultCat");
      gramSize = params.getGramSize();
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    } catch (InvalidDatastoreException e) {
      log.error(e.toString(), e);
    }
  }
}
