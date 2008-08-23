package org.apache.mahout.classifier.bayes;

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

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class WikipediaDatasetCreatorMapper extends MapReduceBase implements
    Mapper<Text, Text, Text, Text> {

  static Set<String> countries = null;

  
  @SuppressWarnings("deprecation")
  public void map(Text key, Text value,
      OutputCollector<Text, Text> output, Reporter reporter)
      throws IOException {
    String document = value.toString();
    Analyzer analyzer = new StandardAnalyzer();
    StringBuilder contents = new StringBuilder();
    
    
    HashSet<String> categories = new HashSet<String>(findAllCategories(document));
    
    String country = getCountry(categories);
    
    if(!country.equals("Unknown")){
      document = StringEscapeUtils.unescapeHtml(document.replaceFirst("<text xml:space=\"preserve\">", "").replaceAll("</text>", ""));
      TokenStream stream = analyzer.tokenStream(country, new StringReader(document));
      while(true){
        Token token = stream.next();
        if(token==null) break;
        contents.append(token.termText()).append(" ");
      }
      //System.err.println(country+"\t"+contents.toString());
      output.collect(new Text(country.replace(" ","_")), new Text(contents.toString()));
    }
  }
  
  public String getCountry(Set<String> categories)
  {
    for(String category : categories)
    {
      for(String country: countries){        
        if(category.indexOf(country)!=-1){
          return country;
          
        }
      }      
    }
    return "Unknown";
  }
  
  public List<String> findAllCategories(String document){
    List<String> categories =  new ArrayList<String>();
    int startIndex = 0;
    int categoryIndex;
    
    while((categoryIndex = document.indexOf("[[Category:", startIndex))!=-1)
    {
      categoryIndex+=11;
      int endIndex = document.indexOf("]]", categoryIndex);
      if(endIndex>=document.length() || endIndex < 0) break;
      String category = document.substring(categoryIndex, endIndex);
      categories.add(category);
      startIndex = endIndex;
    }
    
    return categories;
  }
  
  @Override
  public void configure(JobConf job) {
    try
    {
      if(countries ==null){
        countries = new HashSet<String>();

        DefaultStringifier<Set<String>> setStringifier = new DefaultStringifier<Set<String>>(job,GenericsUtil.getClass(countries));

        String countriesString = setStringifier.toString(countries);  
        countriesString = job.get("wikipedia.countries", countriesString);
        
        countries = setStringifier.fromString(countriesString);
        
      }
    }
    catch(IOException ex){
      
      ex.printStackTrace();
    }
  }
}
