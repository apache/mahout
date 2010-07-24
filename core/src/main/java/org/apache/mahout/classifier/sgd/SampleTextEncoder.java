/*
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

package org.apache.mahout.classifier.sgd;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.StringReader;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: Jul 11, 2010 Time: 7:12:14 PM To change this
 * template use File | Settings | File Templates.
 */
public class SampleTextEncoder {
  public static void main(String[] args) throws IOException {
    RecordValueEncoder encoder = new StaticWordValueEncoder("text");
    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);

    StringReader in = new StringReader("text to magically vectorize");
    TokenStream ts = analyzer.tokenStream("body", in);
    TermAttribute termAtt = ts.addAttribute(TermAttribute.class);

    Vector v1 = new RandomAccessSparseVector(20);
    while (ts.incrementToken()) {
      char[] termBuffer = termAtt.termBuffer();
      int termLen = termAtt.termLength();
      String w = new String(termBuffer, 0, termLen);
      encoder.addToVector(w, 1, v1);
    }
    System.out.printf("%s\n", new SequentialAccessSparseVector(v1));

    Vector v2 = new RandomAccessSparseVector(20);
    encoder.addToVector("text", 1, v2);
    System.out.printf("%s %s\n", "text", v2);

    Vector v3 = new RandomAccessSparseVector(20);
    encoder.addToVector("magically", 1, v3);
    System.out.printf("%s %s\n", "magically", v3);

    Vector v4 = new RandomAccessSparseVector(20);
    encoder.addToVector("vectorize", 1, v4);
    System.out.printf("%s %s\n", "vectorize", v4);
  }
}
