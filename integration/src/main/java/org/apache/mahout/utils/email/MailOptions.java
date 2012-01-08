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

package org.apache.mahout.utils.email;

import java.io.File;
import java.nio.charset.Charset;
import java.util.Map;
import java.util.regex.Pattern;

public class MailOptions {

  public static final String FROM = "FROM";
  public static final String TO = "TO";
  public static final String REFS = "REFS";
  public static final String SUBJECT = "SUBJECT";
  public static final Pattern DEFAULT_QUOTED_TEXT = Pattern.compile("^(\\||>)");

  private boolean stripQuotedText;
  private File input;
  private String outputDir;
  private String prefix;
  private int chunkSize;
  private Charset charset;
  private String separator;
  private String bodySeparator = "\n";
  private boolean includeBody;
  private Pattern[] patternsToMatch;
  //maps FROM, TO, REFS, SUBJECT, etc. to the order they appear in patternsToMatch.  See MailToRecMapper
  private Map<String, Integer> patternOrder;

  //the regular expression to use for identifying quoted text.
  private Pattern quotedTextPattern = DEFAULT_QUOTED_TEXT;

  public File getInput() {
    return input;
  }

  public void setInput(File input) {
    this.input = input;
  }

  public String getOutputDir() {
    return outputDir;
  }

  public void setOutputDir(String outputDir) {
    this.outputDir = outputDir;
  }

  public String getPrefix() {
    return prefix;
  }

  public void setPrefix(String prefix) {
    this.prefix = prefix;
  }

  public int getChunkSize() {
    return chunkSize;
  }

  public void setChunkSize(int chunkSize) {
    this.chunkSize = chunkSize;
  }

  public Charset getCharset() {
    return charset;
  }

  public void setCharset(Charset charset) {
    this.charset = charset;
  }

  public String getSeparator() {
    return separator;
  }

  public void setSeparator(String separator) {
    this.separator = separator;
  }

  public String getBodySeparator() {
    return bodySeparator;
  }

  public void setBodySeparator(String bodySeparator) {
    this.bodySeparator = bodySeparator;
  }

  public boolean isIncludeBody() {
    return includeBody;
  }

  public void setIncludeBody(boolean includeBody) {
    this.includeBody = includeBody;
  }

  public Pattern[] getPatternsToMatch() {
    return patternsToMatch;
  }

  public void setPatternsToMatch(Pattern[] patternsToMatch) {
    this.patternsToMatch = patternsToMatch;
  }

  public Map<String, Integer> getPatternOrder() {
    return patternOrder;
  }

  public void setPatternOrder(Map<String, Integer> patternOrder) {
    this.patternOrder = patternOrder;
  }

  /**
   *
   * @return true if we should strip out quoted email text
   */
  public boolean isStripQuotedText() {
    return stripQuotedText;
  }

  /**
   *
   * @param stripQuotedText if true, then strip off quoted text, such as lines starting with | or >
   */
  public void setStripQuotedText(boolean stripQuotedText) {
    this.stripQuotedText = stripQuotedText;
  }

  public Pattern getQuotedTextPattern() {
    return quotedTextPattern;
  }

  /**
   * @see #setStripQuotedText(boolean)
   *
   * @param quotedTextPattern The {@link java.util.regex.Pattern} to use to identify lines that are quoted text.  Default is | and >
   */
  public void setQuotedTextPattern(Pattern quotedTextPattern) {
    this.quotedTextPattern = quotedTextPattern;
  }
}
