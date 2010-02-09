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

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

/**
 * Runs pass 1 of the Collocation discovery job on input of
 * SequeceFile<Text,Text>, where the key is a document id and the value is the
 * document contents. . Delegates to NGramCollector to perform tokenization,
 * ngram-creation and output collection.
 * 
 * @see org.apache.mahout.text.SequenceFilesFromDirectory
 * @see org.apache.mahout.utils.nlp.collocations.llr.colloc.NGramCollector
 */
public class CollocMapper extends MapReduceBase implements
    Mapper<Text,Text,Gram,Gram> {
  
  private final NGramCollector ngramCollector;
  
  public CollocMapper() {
    ngramCollector = new NGramCollector();
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    ngramCollector.configure(job);
  }
  
  /**
   * Collocation finder: pass 1 map phase.
   * 
   * receives full documents in value and passes these to
   * NGramCollector.collectNGrams.
   * 
   * @see org.apache.mahout.utils.nlp.collocations.llr.colloc.NGramCollector#collectNgrams(Reader,
   *      OutputCollector, Reporter)
   */
  @Override
  public void map(Text key,
                  Text value,
                  OutputCollector<Gram,Gram> collector,
                  Reporter reporter) throws IOException {
    
    Reader r = new StringReader(value.toString());
    ngramCollector.collectNgrams(r, collector, reporter);
    
  }
}
