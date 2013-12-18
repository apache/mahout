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

import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.StopwordAnalyzerBase;
import org.apache.lucene.util.Version;

/**
 * Custom Lucene Analyzer designed for aggressive feature reduction
 * for clustering the ASF Mail Archives using an extended set of
 * stop words, excluding non-alpha-numeric tokens, and porter stemming.
 */
public final class MailArchivesClusteringAnalyzer extends StopwordAnalyzerBase {
  private static final Version LUCENE_VERSION = Version.LUCENE_46;
  
  // extended set of stop words composed of common mail terms like "hi",
  // HTML tags, and Java keywords asmany of the messages in the archives
  // are subversion check-in notifications
    
  private static final CharArraySet STOP_SET = new CharArraySet(LUCENE_VERSION, Arrays.asList(
    "3d","7bit","a0","about","above","abstract","across","additional","after",
    "afterwards","again","against","align","all","almost","alone","along",
    "already","also","although","always","am","among","amongst","amoungst",
    "amount","an","and","another","any","anybody","anyhow","anyone","anything",
    "anyway","anywhere","are","arial","around","as","ascii","assert","at",
    "back","background","base64","bcc","be","became","because","become","becomes",
    "becoming","been","before","beforehand","behind","being","below","beside",
    "besides","between","beyond","bgcolor","blank","blockquote","body","boolean",
    "border","both","br","break","but","by","can","cannot","cant","case","catch",
    "cc","cellpadding","cellspacing","center","char","charset","cheers","class",
    "co","color","colspan","com","con","const","continue","could","couldnt",
    "cry","css","de","dear","default","did","didnt","different","div","do",
    "does","doesnt","done","dont","double","down","due","during","each","eg",
    "eight","either","else","elsewhere","empty","encoding","enough","enum",
    "etc","eu","even","ever","every","everyone","everything","everywhere",
    "except","extends","face","family","few","ffffff","final","finally","float",
    "font","for","former","formerly","fri","from","further","get","give","go",
    "good","got","goto","gt","h1","ha","had","has","hasnt","have","he","head",
    "height","hello","helvetica","hence","her","here","hereafter","hereby",
    "herein","hereupon","hers","herself","hi","him","himself","his","how",
    "however","hr","href","html","http","https","id","ie","if","ill","im",
    "image","img","implements","import","in","inc","instanceof","int","interface",
    "into","is","isnt","iso-8859-1","it","its","itself","ive","just","keep",
    "last","latter","latterly","least","left","less","li","like","long","look",
    "lt","ltd","mail","mailto","many","margin","may","me","meanwhile","message",
    "meta","might","mill","mine","mon","more","moreover","most","mostly","mshtml",
    "mso","much","must","my","myself","name","namely","native","nbsp","need",
    "neither","never","nevertheless","new","next","nine","no","nobody","none",
    "noone","nor","not","nothing","now","nowhere","null","of","off","often",
    "ok","on","once","only","onto","or","org","other","others","otherwise",
    "our","ours","ourselves","out","over","own","package","pad","per","perhaps",
    "plain","please","pm","printable","private","protected","public","put",
    "quot","quote","r1","r2","rather","re","really","regards","reply","return",
    "right","said","same","sans","sat","say","saying","see","seem","seemed",
    "seeming","seems","serif","serious","several","she","short","should","show",
    "side","since","sincere","six","sixty","size","so","solid","some","somehow",
    "someone","something","sometime","sometimes","somewhere","span","src",
    "static","still","strictfp","string","strong","style","stylesheet","subject",
    "such","sun","super","sure","switch","synchronized","table","take","target",
    "td","text","th","than","thanks","that","the","their","them","themselves",
    "then","thence","there","thereafter","thereby","therefore","therein","thereupon",
    "these","they","thick","thin","think","third","this","those","though",
    "three","through","throughout","throw","throws","thru","thu","thus","tm",
    "to","together","too","top","toward","towards","tr","transfer","transient",
    "try","tue","type","ul","un","under","unsubscribe","until","up","upon",
    "us","use","used","uses","using","valign","verdana","very","via","void",
    "volatile","want","was","we","wed","weight","well","were","what","whatever",
    "when","whence","whenever","where","whereafter","whereas","whereby","wherein",
    "whereupon","wherever","whether","which","while","whither","who","whoever",
    "whole","whom","whose","why","width","will","with","within","without",
    "wont","would","wrote","www","yes","yet","you","your","yours","yourself",
    "yourselves"
  ), false);

  // Regex used to exclude non-alpha-numeric tokens
  private static final Pattern ALPHA_NUMERIC = Pattern.compile("^[a-z][a-z0-9_]+$");
  private static final Matcher MATCHER = ALPHA_NUMERIC.matcher("");

  public MailArchivesClusteringAnalyzer() {
    super(LUCENE_VERSION, STOP_SET);
  }

  public MailArchivesClusteringAnalyzer(CharArraySet stopSet) {
    super(LUCENE_VERSION, stopSet);

  }
  
  @Override
  protected TokenStreamComponents createComponents(String fieldName, Reader reader) {
    Tokenizer tokenizer = new StandardTokenizer(LUCENE_VERSION, reader);
    TokenStream result = new StandardFilter(LUCENE_VERSION, tokenizer);
    result = new LowerCaseFilter(LUCENE_VERSION, result);
    result = new ASCIIFoldingFilter(result);
    result = new AlphaNumericMaxLengthFilter(result);
    result = new StopFilter(LUCENE_VERSION, result, STOP_SET);
    result = new PorterStemFilter(result);
    return new TokenStreamComponents(tokenizer, result);
  }

  /**
   * Matches alpha-numeric tokens between 2 and 40 chars long.
   */
  static class AlphaNumericMaxLengthFilter extends TokenFilter {
    private final CharTermAttribute termAtt;
    private final char[] output = new char[28];

    AlphaNumericMaxLengthFilter(TokenStream in) {
      super(in);
      termAtt = addAttribute(CharTermAttribute.class);
    }

    @Override
    public final boolean incrementToken() throws IOException {
      // return the first alpha-numeric token between 2 and 40 length
      while (input.incrementToken()) {
        int length = termAtt.length();
        if (length >= 2 && length <= 28) {
          char[] buf = termAtt.buffer();
          int at = 0;
          for (int c = 0; c < length; c++) {
            char ch = buf[c];
            if (ch != '\'') {
              output[at++] = ch;
            }
          }
          String term = new String(output, 0, at);
          MATCHER.reset(term);
          if (MATCHER.matches() && !term.startsWith("a0")) {
            termAtt.setEmpty();
            termAtt.append(term);
            return true;
          }
        }
      }
      return false;
    }
  }
}
