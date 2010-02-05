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

package org.apache.mahout.utils.vectors.text;

import java.io.IOException;
import java.nio.charset.Charset;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.utils.vectors.text.document.SequenceFileTokenizerMapper;

/**
 * This class converts a set of input documents in the sequence file format of
 * {@link StringTuple}s.The {@link SequenceFile} input should have a
 * {@link Text} key containing the unique document identifier and a {@link Text}
 * value containing the whole document. The document should be stored in UTF-8
 * encoding which is recognizable by hadoop. It uses the given {@link Analyzer}
 * to process the document into {@link org.apache.lucene.analysis.Token}s.
 * 
 */
public final class DocumentProcessor {
  
  public static final String ANALYZER_CLASS = "analyzer.class";
  
  public static final Charset CHARSET = Charset.forName("UTF-8");
  
  /**
   * Cannot be initialized. Use the static functions
   */
  private DocumentProcessor() {

  }
  
  /**
   * Convert the input documents into token array using the {@link StringTuple}
   * The input documents has to be in the {@link SequenceFile} format
   * 
   * @param input
   *          input directory of the documents in {@link SequenceFile} format
   * @param output
   *          output directory were the {@link StringTuple} token array of each
   *          document has to be created
   * @param analyzerClass
   *          The Lucene {@link Analyzer} for tokenizing the UTF-8 text
   * @throws IOException
   */
  public static void tokenizeDocuments(String input,
                                       Class<? extends Analyzer> analyzerClass,
                                       String output) throws IOException {
    
    Configurable client = new JobClient();
    JobConf conf = new JobConf(DocumentProcessor.class);
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    // this conf parameter needs to be set enable serialisation of conf values
    
    conf.set(ANALYZER_CLASS, analyzerClass.getName());
    conf.setJobName("DocumentProcessor::DocumentTokenizer: input-folder: "
                    + input);
    
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(StringTuple.class);
    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setMapperClass(SequenceFileTokenizerMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setNumReduceTasks(0);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    
    client.setConf(conf);
    JobClient.runJob(conf);
  }
}
