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

package org.apache.mahout.utils.nlp.collocations.llr;

import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.HEAD;
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.TAIL;
import static org.apache.mahout.utils.nlp.collocations.llr.NGramCollector.Count.NGRAM_TOTAL;

import java.io.IOException;
import java.io.Reader;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.util.Version;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Performs tokenization, ngram generation + collection for the first pass of
 * the LLR collocation discovery job. Factors this code out of the mappers so
 * that different input formats can be supported.
 * 
 * @see org.apache.mahout.utils.nlp.collocations.llr.colloc.CollocMapperTextFile
 */
public class NGramCollector {
  
  public static final String ANALYZER_CLASS = "analyzerClass";
  public static final String MAX_SHINGLE_SIZE = "maxShingleSize";
  
  public static enum Count {
    NGRAM_TOTAL;
  }
  
  private static final Logger log = LoggerFactory
      .getLogger(NGramCollector.class);
  
  /**
   * An analyzer to perform tokenization. A ShingleFilter will be wrapped around
   * its output to create ngrams
   */
  private Analyzer a;
  
  /** max size of shingles (ngrams) to create */
  private int maxShingleSize;
  
  public NGramCollector() {}
  
  /**
   * Configure the NGramCollector.
   * 
   * Reads NGramCollector.ANALYZER_CLASS and instantiates that class if it is
   * provided. Otherwise a lucene StandardAnalyzer will be used that is set to
   * be compatible to LUCENE_24.
   * 
   * Reads NGramCollector.MAX_SHINGLE_SIZE and uses this as the parameter to the
   * ShingleFilter.
   * 
   * @param job
   */
  public void configure(JobConf job) {
    this.a = null;
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      String analyzerClass = job.get(NGramCollector.ANALYZER_CLASS);
      if (analyzerClass != null) {
        Class<?> cl = ccl.loadClass(analyzerClass);
        a = (Analyzer) cl.newInstance();
      }
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }
    
    if (this.a == null) {
      // No analyzer specified. Use the LUCENE_24 analzer here because
      // it does not preserve stop word positions.
      this.a = new StandardAnalyzer(Version.LUCENE_24);
    }
    
    this.maxShingleSize = job.getInt(NGramCollector.MAX_SHINGLE_SIZE, 2);
    
    if (log.isInfoEnabled()) {
      log.info("Analyzer is {}", this.a.toString());
      log.info("Max Ngram size is {}", this.maxShingleSize);
    }
  }
  
  /**
   * Receives a document and uses a lucene analyzer to tokenize them. The
   * ShingleFilter delivers ngrams of the appropriate size which aren then
   * decomposed into head and tail subgrams which are collected in the following
   * manner
   * 
   * k:h_subgram v:ngram k:t_subgram v:ngram
   * 
   * The 'h_' or 't_' prefix is used to specify whether the subgram in question
   * is the head or tail of the ngram. In this implementation the head of the
   * ngram is a (n-1)gram, and the tail is a (1)gram.
   * 
   * For example, given 'click and clack' and an ngram length of 3: k:'h_click
   * and' v:'clack and clack' k;'t_clack' v:'click and clack'
   * 
   * Also counts the total number of ngrams encountered and adds it to the
   * counter CollocDriver.Count.NGRAM_TOTAL
   * 
   * @param r
   *          The reader to read input from -- used to create a tokenstream from
   *          the analyzer
   * 
   * @param collector
   *          The collector to send output to
   * 
   * @param reporter
   *          Used to deliver the final ngram-count.
   * 
   * @throws IOException
   *           if there's a problem with the ShingleFilter reading data or the
   *           collector collecting output.
   */
  public void collectNgrams(Reader r,
                            OutputCollector<Gram,Gram> collector,
                            Reporter reporter) throws IOException {
    TokenStream st = a.tokenStream("text", r);
    ShingleFilter sf = new ShingleFilter(st, maxShingleSize);
    
    sf.reset();
    int count = 0; // ngram count
    
    do {
      String term = ((TermAttribute) sf.getAttribute(TermAttribute.class))
          .term();
      String type = ((TypeAttribute) sf.getAttribute(TypeAttribute.class))
          .type();
      
      if ("shingle".equals(type)) {
        count++;
        Gram ngram = new Gram(term);
        
        // obtain components, the leading (n-1)gram and the trailing unigram.
        int i = term.lastIndexOf(' ');
        if (i != -1) {
          collector.collect(new Gram(term.substring(0, i), HEAD), ngram);
          collector.collect(new Gram(term.substring(i + 1), TAIL), ngram);
        }
      }
    } while (sf.incrementToken());
    
    reporter.incrCounter(NGRAM_TOTAL, count);
    
    sf.end();
    sf.close();
    r.close();
  }
}
