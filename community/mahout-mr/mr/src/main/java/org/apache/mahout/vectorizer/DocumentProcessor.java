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

package org.apache.mahout.vectorizer;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.vectorizer.document.SequenceFileTokenizerMapper;

/**
 * This class converts a set of input documents in the sequence file format of {@link StringTuple}s.The
 * {@link org.apache.hadoop.io.SequenceFile} input should have a {@link Text} key
 * containing the unique document identifier and a
 * {@link Text} value containing the whole document. The document should be stored in UTF-8 encoding which is
 * recognizable by hadoop. It uses the given {@link Analyzer} to process the document into
 * {@link org.apache.lucene.analysis.Token}s.
 * 
 */
public final class DocumentProcessor {
  
  public static final String TOKENIZED_DOCUMENT_OUTPUT_FOLDER = "tokenized-documents";
  public static final String ANALYZER_CLASS = "analyzer.class";

  /**
   * Cannot be initialized. Use the static functions
   */
  private DocumentProcessor() {

  }
  
  /**
   * Convert the input documents into token array using the {@link StringTuple} The input documents has to be
   * in the {@link org.apache.hadoop.io.SequenceFile} format
   * 
   * @param input
   *          input directory of the documents in {@link org.apache.hadoop.io.SequenceFile} format
   * @param output
   *          output directory were the {@link StringTuple} token array of each document has to be created
   * @param analyzerClass
   *          The Lucene {@link Analyzer} for tokenizing the UTF-8 text
   */
  public static void tokenizeDocuments(Path input,
                                       Class<? extends Analyzer> analyzerClass,
                                       Path output,
                                       Configuration baseConf)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration(baseConf);
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization"); 
    conf.set(ANALYZER_CLASS, analyzerClass.getName());
    
    Job job = new Job(conf);
    job.setJobName("DocumentProcessor::DocumentTokenizer: input-folder: " + input);
    job.setJarByClass(DocumentProcessor.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(StringTuple.class);
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    
    job.setMapperClass(SequenceFileTokenizerMapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setNumReduceTasks(0);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    HadoopUtil.delete(conf, output);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }

  }
}
