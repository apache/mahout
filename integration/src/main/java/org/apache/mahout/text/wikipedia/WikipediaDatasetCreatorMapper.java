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

package org.apache.mahout.text.wikipedia;

import java.io.IOException;
import java.io.StringReader;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.ClassUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Maps over Wikipedia xml format and output all document having the category listed in the input category
 * file
 * 
 */
public class WikipediaDatasetCreatorMapper extends Mapper<LongWritable, Text, Text, Text> {

  private static final Logger log = LoggerFactory.getLogger(WikipediaDatasetCreatorMapper.class);

  private static final Pattern SPACE_NON_ALPHA_PATTERN = Pattern.compile("[\\s\\W]");
  private static final Pattern OPEN_TEXT_TAG_PATTERN = Pattern.compile("<text xml:space=\"preserve\">");
  private static final Pattern CLOSE_TEXT_TAG_PATTERN = Pattern.compile("</text>");

  private List<String> inputCategories;
  private List<Pattern> inputCategoryPatterns;
  private boolean exactMatchOnly;
  private Analyzer analyzer;

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    String document = value.toString();
    document = StringEscapeUtils.unescapeHtml4(CLOSE_TEXT_TAG_PATTERN.matcher(
        OPEN_TEXT_TAG_PATTERN.matcher(document).replaceFirst("")).replaceAll(""));
    String catMatch = findMatchingCategory(document);
    if (!"Unknown".equals(catMatch)) {
      StringBuilder contents = new StringBuilder(1000);
      TokenStream stream = analyzer.tokenStream(catMatch, new StringReader(document));
      CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
      stream.reset();
      while (stream.incrementToken()) {
        contents.append(termAtt.buffer(), 0, termAtt.length()).append(' ');
      }
      context.write(
          new Text(SPACE_NON_ALPHA_PATTERN.matcher(catMatch).replaceAll("_")),
          new Text(contents.toString()));
      stream.end();
      Closeables.close(stream, true);
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);

    Configuration conf = context.getConfiguration();

    if (inputCategories == null) {
      Set<String> newCategories = Sets.newHashSet();
      DefaultStringifier<Set<String>> setStringifier =
          new DefaultStringifier<Set<String>>(conf, GenericsUtil.getClass(newCategories));
      String categoriesStr = conf.get("wikipedia.categories", setStringifier.toString(newCategories));
      Set<String> inputCategoriesSet = setStringifier.fromString(categoriesStr);
      inputCategories = Lists.newArrayList(inputCategoriesSet);
      inputCategoryPatterns = Lists.newArrayListWithCapacity(inputCategories.size());
      for (String inputCategory : inputCategories) {
        inputCategoryPatterns.add(Pattern.compile(".*\\b" + inputCategory + "\\b.*"));
      }

    }

    exactMatchOnly = conf.getBoolean("exact.match.only", false);

    if (analyzer == null) {
      String analyzerStr = conf.get("analyzer.class", WikipediaAnalyzer.class.getName());
      analyzer = ClassUtils.instantiateAs(analyzerStr, Analyzer.class);
    }

    log.info("Configure: Input Categories size: {} Exact Match: {} Analyzer: {}",
             inputCategories.size(), exactMatchOnly, analyzer.getClass().getName());
  }

  private String findMatchingCategory(String document) {
    int startIndex = 0;
    int categoryIndex;
    while ((categoryIndex = document.indexOf("[[Category:", startIndex)) != -1) {
      categoryIndex += 11;
      int endIndex = document.indexOf("]]", categoryIndex);
      if (endIndex >= document.length() || endIndex < 0) {
        break;
      }
      String category = document.substring(categoryIndex, endIndex).toLowerCase(Locale.ENGLISH).trim();
      // categories.add(category.toLowerCase());
      if (exactMatchOnly && inputCategories.contains(category)) {
        return category;
      }
      if (!exactMatchOnly) {
        for (int i = 0; i < inputCategories.size(); i++) {
          String inputCategory = inputCategories.get(i);
          Pattern inputCategoryPattern = inputCategoryPatterns.get(i);
          if (inputCategoryPattern.matcher(category).matches()) { // inexact match with word boundary.
            return inputCategory;
          }
        }
      }
      startIndex = endIndex;
    }
    return "Unknown";
  }
}
