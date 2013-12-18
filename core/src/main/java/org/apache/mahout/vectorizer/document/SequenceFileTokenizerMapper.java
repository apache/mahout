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

package org.apache.mahout.vectorizer.document;

import java.io.IOException;
import java.io.StringReader;

import com.google.common.io.Closeables;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;

import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.lucene.AnalyzerUtils;
import org.apache.mahout.vectorizer.DocumentProcessor;

/**
 * Tokenizes a text document and outputs tokens in a StringTuple
 */
public class SequenceFileTokenizerMapper extends Mapper<Text, Text, Text, StringTuple> {

  private Analyzer analyzer;

  @Override
  protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
    TokenStream stream = analyzer.tokenStream(key.toString(), new StringReader(value.toString()));
    CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
    stream.reset();
    StringTuple document = new StringTuple();
    while (stream.incrementToken()) {
      if (termAtt.length() > 0) {
        document.add(new String(termAtt.buffer(), 0, termAtt.length()));
      }
    }
    stream.end();
    Closeables.close(stream, true);
    context.write(key, document);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    String analyzerClassName = context.getConfiguration().get(DocumentProcessor.ANALYZER_CLASS,
            StandardAnalyzer.class.getName());
    try {
      analyzer = AnalyzerUtils.createAnalyzer(analyzerClassName);
    } catch (ClassNotFoundException e) {
      throw new IOException("Unable to create analyzer: " + analyzerClassName, e);
    }
  }
}
