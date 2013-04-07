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
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.collect.Sets;
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

    if (!all) {
      String catMatch = findMatchingCategory(document);
      if ("Unknown".equals(catMatch)) {
        return;
      }
    }
    document = StringEscapeUtils.unescapeHtml4(document);
    context.write(new Text(SPACE_NON_ALPHA_PATTERN.matcher(title).replaceAll("_")), new Text(document));
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
      inputCategories = setStringifier.fromString(categoriesStr);
    }
    exactMatchOnly = conf.getBoolean("exact.match.only", false);
    all = conf.getBoolean("all.files", true);
    log.info("Configure: Input Categories size: {} All: {} Exact Match: {}",
             inputCategories.size(), all, exactMatchOnly);
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
        return category;
      }
      if (!exactMatchOnly) {
        for (String inputCategory : inputCategories) {
          if (category.contains(inputCategory)) { // we have an inexact match
            return inputCategory;
          }
        }
      }
      startIndex = endIndex;
    }
    return "Unknown";
  }
}
