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
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.GenericsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maps over Wikipedia xml format and output all document having the category listed in the input category
 * file
 * 
 */
public class WikipediaMapper extends Mapper<LongWritable, Text, Text, Text> {

  private static final Logger log = LoggerFactory.getLogger(WikipediaMapper.class);

  private static final Pattern SPACE_NON_ALPHA_PATTERN = Pattern.compile("[\\s]");

  private static final String START_DOC = "<text xml:space=\"preserve\">";

  private static final String END_DOC = "</text>";

  private static final Pattern TITLE = Pattern.compile("<title>(.*)<\\/title>");

  private static final String REDIRECT = "<redirect />";

  private Set<String> inputCategories;

  private boolean exactMatchOnly;

  private boolean all;

  private boolean removeLabels;

  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

    String content = value.toString();
    if (content.contains(REDIRECT)) {
      return;
    }
    String document;
    String title;
    try {
      document = getDocument(content);
      title = getTitle(content);
    } catch (RuntimeException e) {
      // TODO: reporter.getCounter("Wikipedia", "Parse errors").increment(1);
      return;
    }

    String catMatch = findMatchingCategory(document);
    if (!all) {
      if ("Unknown".equals(catMatch)) {
        return;
      }
    }

    document = StringEscapeUtils.unescapeHtml4(document);    
    if (removeLabels) {
      document = removeCategoriesFromText(document);
      // Reject documents with malformed tags
      if (document == null) {
        return;
      }
    }

    // write out in Bayes input style: key: /Category/document_name
    String category = "/" + catMatch.toLowerCase(Locale.ENGLISH) + "/" +
        SPACE_NON_ALPHA_PATTERN.matcher(title).replaceAll("_");

    context.write(new Text(category), new Text(document));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
 
    Set<String> newCategories = new HashSet<>();
    DefaultStringifier<Set<String>> setStringifier =
          new DefaultStringifier<>(conf, GenericsUtil.getClass(newCategories));

    String categoriesStr = conf.get("wikipedia.categories");
    inputCategories = setStringifier.fromString(categoriesStr);
    exactMatchOnly = conf.getBoolean("exact.match.only", false);
    all = conf.getBoolean("all.files", false);
    removeLabels = conf.getBoolean("remove.labels",false);
    log.info("Configure: Input Categories size: {} All: {} Exact Match: {} Remove Labels from Text: {}",
            inputCategories.size(), all, exactMatchOnly, removeLabels);
  }

  private static String getDocument(String xml) {
    int start = xml.indexOf(START_DOC) + START_DOC.length();
    int end = xml.indexOf(END_DOC, start);
    return xml.substring(start, end);
  }

  private static String getTitle(CharSequence xml) {
    Matcher m = TITLE.matcher(xml);
    return m.find() ? m.group(1) : "";
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
      if (exactMatchOnly && inputCategories.contains(category)) {
        return category.toLowerCase(Locale.ENGLISH);
      }
      if (!exactMatchOnly) {
        for (String inputCategory : inputCategories) {
          if (category.contains(inputCategory)) { // we have an inexact match
            return inputCategory.toLowerCase(Locale.ENGLISH);
          }
        }
      }
      startIndex = endIndex;
    }
    return "Unknown";
  }

  private String removeCategoriesFromText(String document) {
    int startIndex = 0;
    int categoryIndex;
    try {
      while ((categoryIndex = document.indexOf("[[Category:", startIndex)) != -1) {
        int endIndex = document.indexOf("]]", categoryIndex);
        if (endIndex >= document.length() || endIndex < 0) {
          break;
        }
        document = document.replace(document.substring(categoryIndex, endIndex + 2), "");
        if (categoryIndex < document.length()) {
          startIndex = categoryIndex;
        } else {
          break;
        }
      }
    } catch(StringIndexOutOfBoundsException e) {
      return null;
    }
    return document;
  }
}
