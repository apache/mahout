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

package org.apache.mahout.classifier.bayes;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.mahout.analysis.WikipediaAnalyzer;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

public class WikipediaDatasetCreatorMapper extends MapReduceBase implements
    Mapper<LongWritable, Text, Text, Text> {

  private static final Logger log = LoggerFactory.getLogger(WikipediaDatasetCreatorMapper.class);
  private static final Pattern SPACE_NON_ALPHA_PATTERN = Pattern.compile("[\\s\\W]");
  private static final Pattern OPEN_TEXT_TAG_PATTERN = Pattern.compile("<text xml:space=\"preserve\">");
  private static final Pattern CLOSE_TEXT_TAG_PATTERN = Pattern.compile("</text>");

  private Set<String> inputCategories = null;
  private boolean exactMatchOnly = false;
  private Analyzer analyzer;

  @Override
  public void map(LongWritable key, Text value,
      OutputCollector<Text, Text> output, Reporter reporter)
      throws IOException {

    StringBuilder contents = new StringBuilder();
    String document = value.toString();
    String catMatch = findMatchingCategory(document);
    
    if(!catMatch.equals("Unknown")){
      document = StringEscapeUtils.unescapeHtml(CLOSE_TEXT_TAG_PATTERN.matcher(OPEN_TEXT_TAG_PATTERN.matcher(document).replaceFirst("")).replaceAll(""));
      TokenStream stream = analyzer.tokenStream(catMatch, new StringReader(document));
      Token token = new Token();
      while((token = stream.next(token)) != null){
        contents.append(token.termBuffer(), 0, token.termLength()).append(' ');
      }
      output.collect(new Text(SPACE_NON_ALPHA_PATTERN.matcher(catMatch).replaceAll("_")), new Text(contents.toString()));
    }
  }

  private String findMatchingCategory(String document){
    int startIndex = 0;
    int categoryIndex;
    while((categoryIndex = document.indexOf("[[Category:", startIndex))!=-1)
    {
      categoryIndex+=11;
      int endIndex = document.indexOf("]]", categoryIndex);
      if (endIndex >= document.length() || endIndex < 0) {
        break;
      }
      String category = document.substring(categoryIndex, endIndex).toLowerCase().trim();
      //categories.add(category.toLowerCase());
      if (exactMatchOnly && inputCategories.contains(category)){
        return category;
      } else if (exactMatchOnly == false){
        for (String inputCategory : inputCategories) {
          if (category.contains(inputCategory)){//we have an inexact match
            return inputCategory;
          }
        }
      }
      startIndex = endIndex;
    }
    return "Unknown";
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      if (inputCategories == null){
        Set<String> newCategories = new HashSet<String>();

        DefaultStringifier<Set<String>> setStringifier =
            new DefaultStringifier<Set<String>>(job,GenericsUtil.getClass(newCategories));

        String categoriesStr = setStringifier.toString(newCategories);
        categoriesStr = job.get("wikipedia.categories", categoriesStr);
        inputCategories = setStringifier.fromString(categoriesStr);

      }
      exactMatchOnly = job.getBoolean("exact.match.only", false);
      if (analyzer == null){
        String analyzerStr = job.get("analyzer.class", WikipediaAnalyzer.class.getName());
        Class<? extends Analyzer> analyzerClass = (Class<? extends Analyzer>) Class.forName(analyzerStr);
        analyzer = analyzerClass.newInstance();
      }
    } catch(IOException ex){
      throw new IllegalStateException(ex);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    log.info("Configure: Input Categories size: " + inputCategories.size() + " Exact Match: " + exactMatchOnly +
             " Analyzer: " + analyzer.getClass().getName());
  }
}
