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

package org.apache.mahout.utils.regex;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.ClassUtils;

import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;

public class RegexMapper extends Mapper<LongWritable, Text, LongWritable, Text> {

  public static final String REGEX = "regex";
  public static final String GROUP_MATCHERS = "regex.groups";
  public static final String TRANSFORMER_CLASS = "transformer.class";
  public static final String FORMATTER_CLASS = "formatter.class";

  private Pattern regex;
  private List<Integer> groupsToKeep;
  private RegexTransformer transformer = RegexUtils.IDENTITY_TRANSFORMER;
  private RegexFormatter formatter = RegexUtils.IDENTITY_FORMATTER;
  public static final String ANALYZER_NAME = "analyzerName";


  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    groupsToKeep = Lists.newArrayList();
    Configuration config = context.getConfiguration();
    String regexStr = config.get(REGEX);
    regex = Pattern.compile(regexStr);
    String[] groups = config.getStrings(GROUP_MATCHERS);
    if (groups != null) {
      for (String group : groups) {
        groupsToKeep.add(Integer.parseInt(group));
      }
    }

    transformer = ClassUtils.instantiateAs(config.get(TRANSFORMER_CLASS, IdentityTransformer.class.getName()),
        RegexTransformer.class);
    String analyzerName = config.get(ANALYZER_NAME);
    if (analyzerName != null && transformer instanceof AnalyzerTransformer) {
      Analyzer analyzer = ClassUtils.instantiateAs(analyzerName, Analyzer.class);
      ((AnalyzerTransformer)transformer).setAnalyzer(analyzer);
    }

    formatter = ClassUtils.instantiateAs(config.get(FORMATTER_CLASS, IdentityFormatter.class.getName()),
        RegexFormatter.class);
  }


  @Override
  protected void map(LongWritable key, Text text, Context context) throws IOException, InterruptedException {
    String result = RegexUtils.extract(text.toString(), regex, groupsToKeep, " ", transformer);
    if (result != null && !result.isEmpty()) {
      String format = formatter.format(result);
      context.write(key, new Text(format));
    }
  }
}
