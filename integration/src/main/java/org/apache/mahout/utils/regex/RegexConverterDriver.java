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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

/**
 * Experimental
 */
public class RegexConverterDriver extends AbstractJob {

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption("regex", "regex",
            "The regular expression to use", true);
    addOption("groupsToKeep", "g",
            "The number of the capturing groups to keep", false);
    addOption("transformerClass", "t",
            "The optional class specifying the Regex Transformer", false);
    addOption("formatterClass", "t",
            "The optional class specifying the Regex Formatter", false);
    addOption(DefaultOptionCreator.analyzerOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Configuration conf = getConf();
    //TODO: How to deal with command line escaping?
    conf.set(RegexMapper.REGEX, getOption("regex")); //
    String gtk = getOption("groupsToKeep");
    if (gtk != null) {
      conf.set(RegexMapper.GROUP_MATCHERS, gtk);
    }
    String trans = getOption("transformerClass");
    if (trans != null) {
      if ("url".equalsIgnoreCase(trans)) {
        trans = URLDecodeTransformer.class.getName();
      }
      conf.set(RegexMapper.TRANSFORMER_CLASS, trans);
    }
    String formatter = getOption("formatterClass");
    if (formatter != null) {
      if ("fpg".equalsIgnoreCase(formatter)) {
        formatter = FPGFormatter.class.getName();
      }
      conf.set(RegexMapper.FORMATTER_CLASS, formatter);
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    Class<? extends Analyzer> analyzerClass = getAnalyzerClassFromOption();
    if (analyzerClass != null) {
      conf.set(RegexMapper.ANALYZER_NAME, analyzerClass.getName());
    }
    Job job = prepareJob(input, output,
            TextInputFormat.class,
            RegexMapper.class,
            LongWritable.class,
            Text.class,
            TextOutputFormat.class);
    boolean succeeded = job.waitForCompletion(true);
    return succeeded ? 0 : -1;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RegexConverterDriver(), args);
  }

}
