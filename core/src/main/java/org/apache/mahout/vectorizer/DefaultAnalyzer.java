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

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.io.Reader;

/**
 *  A subclass of the Lucene StandardAnalyzer that provides a no-argument constructor. 
 *  Used as the default analyzer in many cases where an analyzer is instantiated by
 *  class name by calling a no-arg constructor.
 */
public final class DefaultAnalyzer extends Analyzer {

  private final StandardAnalyzer stdAnalyzer = new StandardAnalyzer(Version.LUCENE_31);

  @Override
  public TokenStream tokenStream(String fieldName, Reader reader) {
    return stdAnalyzer.tokenStream(fieldName, reader);
  }
  
  @Override
  public TokenStream reusableTokenStream(String fieldName, Reader reader) throws IOException {
    return stdAnalyzer.reusableTokenStream(fieldName, reader);
  }
}
