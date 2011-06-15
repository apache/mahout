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
import java.util.regex.Pattern;

import com.google.common.collect.Iterators;
import org.apache.commons.lang.mutable.MutableDouble;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.lucene.IteratorTokenStream;
import org.apache.mahout.math.function.ObjectIntProcedure;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads the input train set(preprocessed using the {@link org.apache.mahout.classifier.BayesFileFormatter}).
 */
public class BayesFeatureMapper extends MapReduceBase implements Mapper<Text,Text,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesFeatureMapper.class);
  
  private static final DoubleWritable ONE = new DoubleWritable(1.0);
  private static final Pattern SPACE_PATTERN = Pattern.compile("[ ]+");

  private int gramSize = 1;

  /**
   * We need to count the number of times we've seen a term with a given label and we need to output that. But
   * this Mapper does more than just outputing the count. It first does weight normalisation. Secondly, it
   * outputs for each unique word in a document value 1 for summing up as the Term Document Frequency. Which
   * later is used to calculate the Idf Thirdly, it outputs for each label the number of times a document was
   * seen(Also used in Idf Calculation)
   * 
   * @param key
   *          The label
   * @param value
   *          the features (all unique) associated w/ this label in stringtuple format
   * @param output
   *          The OutputCollector to write the results to
   * @param reporter
   *          Not used
   */
  @Override
  public void map(Text key,
                  Text value,
                  final OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    final String label = key.toString();
    String[] tokens = SPACE_PATTERN.split(value.toString());
    OpenObjectIntHashMap<String> wordList = new OpenObjectIntHashMap<String>(tokens.length * gramSize);
    
    if (gramSize > 1) {
      ShingleFilter sf = new ShingleFilter(new IteratorTokenStream(Iterators.forArray(tokens)), gramSize);
      do {
        String term = sf.getAttribute(TermAttribute.class).term();
        if (term.length() > 0) {
          if (wordList.containsKey(term)) {
            wordList.put(term, 1 + wordList.get(term));
          } else {
            wordList.put(term, 1);
          }
        }
      } while (sf.incrementToken());
    } else {
      for (String term : tokens) {
        if (wordList.containsKey(term)) {
          wordList.put(term, 1 + wordList.get(term));
        } else {
          wordList.put(term, 1);
        }
      }
    }
    final MutableDouble lengthNormalisationMut = new MutableDouble(0.0);
    wordList.forEachPair(new ObjectIntProcedure<String>() {
      @Override
      public boolean apply(String word, int dKJ) {
        long squared = (long) dKJ * (long) dKJ;
        lengthNormalisationMut.add(squared);
        return true;
      }
    });
    
    final double lengthNormalisation = Math.sqrt(lengthNormalisationMut.doubleValue());
    
    // Output Length Normalized + TF Transformed Frequency per Word per Class
    // Log(1 + D_ij)/SQRT( SIGMA(k, D_kj) )
    wordList.forEachPair(new ObjectIntProcedure<String>() {
      @Override
      public boolean apply(String token, int dKJ) {
        try {
          StringTuple tuple = new StringTuple();
          tuple.add(BayesConstants.WEIGHT);
          tuple.add(label);
          tuple.add(token);
          DoubleWritable f = new DoubleWritable(Math.log(1.0 + dKJ) / lengthNormalisation);
          output.collect(tuple, f);
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
        return true;
      }
    });
    reporter.setStatus("Bayes Feature Mapper: Document Label: " + label);
    
    // Output Document Frequency per Word per Class
    // Corpus Document Frequency (FEATURE_COUNT)
    // Corpus Term Frequency (FEATURE_TF)
    wordList.forEachPair(new ObjectIntProcedure<String>() {
      @Override
      public boolean apply(String token, int dKJ) {
        try {
          StringTuple dfTuple = new StringTuple();
          dfTuple.add(BayesConstants.DOCUMENT_FREQUENCY);
          dfTuple.add(label);
          dfTuple.add(token);
          output.collect(dfTuple, ONE);
          
          StringTuple tokenCountTuple = new StringTuple();
          tokenCountTuple.add(BayesConstants.FEATURE_COUNT);
          tokenCountTuple.add(token);
          output.collect(tokenCountTuple, ONE);
          
          StringTuple tokenTfTuple = new StringTuple();
          tokenTfTuple.add(BayesConstants.FEATURE_TF);
          tokenTfTuple.add(token);
          output.collect(tokenTfTuple, new DoubleWritable(dKJ));
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
        return true;
      }
    });
    
    // output that we have seen the label to calculate the Count of Document per
    // class
    StringTuple labelCountTuple = new StringTuple();
    labelCountTuple.add(BayesConstants.LABEL_COUNT);
    labelCountTuple.add(label);
    output.collect(labelCountTuple, ONE);
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      BayesParameters params = new BayesParameters(job.get("bayes.parameters", ""));
      log.info("Bayes Parameter {}", params.print());
      gramSize = params.getGramSize();
      
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }

}
