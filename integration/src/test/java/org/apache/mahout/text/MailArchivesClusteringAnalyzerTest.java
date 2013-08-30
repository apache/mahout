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

package org.apache.mahout.text;

import java.io.Reader;
import java.io.StringReader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

/**
 * Unit tests for the MailArchivesClusteringAnalyzer text analyzer.
 */
public class MailArchivesClusteringAnalyzerTest extends MahoutTestCase {
  
  @Test
  public void testAnalysis() throws Exception {
    Analyzer analyzer = new MailArchivesClusteringAnalyzer();
    
    String text = "A test message\n"
                  + "atokenthatistoolongtobeusefulforclustertextanalysis\n"
                  + "Mahout is a scalable, machine-learning LIBRARY\n"
                  + "we've added some additional stopwords such as html, mailto, regards\t"
                  + "apache_hadoop provides the foundation for scalability\n"
                  + "www.nabble.com general-help@incubator.apache.org\n"
                  + "public void int protected package";
    Reader reader = new StringReader(text);
    
    // if you change the text above, then you may need to change this as well
    // order matters too
    String[] expectedTokens = {
        "test", "mahout", "scalabl", "machin", "learn", "librari", "weve", "ad",
        "stopword", "apache_hadoop","provid", "foundat", "scalabl"
    };
        
    TokenStream tokenStream = analyzer.tokenStream("test", reader);
    assertNotNull(tokenStream);
    tokenStream.reset();
    CharTermAttribute termAtt = tokenStream.addAttribute(CharTermAttribute.class);
    int e = 0;
    while (tokenStream.incrementToken() && e < expectedTokens.length) {
      assertEquals(expectedTokens[e++], termAtt.toString());
    }
    assertEquals(e, expectedTokens.length);
    tokenStream.end();
    tokenStream.close();
  }
}
