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

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.bayes.common.BayesParameters;
import org.apache.mahout.classifier.bayes.mapreduce.bayes.BayesDriver;
import org.apache.mahout.classifier.bayes.mapreduce.cbayes.CBayesDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Train the Naive Bayes classifier with improved weighting.
 * A properly formatted file for input is one which has one document per line
 * with the first entry as the label and the rest as evidence.
 *
 * @see org.apache.mahout.classifier.BayesFileFormatter
 */
public final class TrainClassifier {
  
  private static final Logger log = LoggerFactory.getLogger(TrainClassifier.class);
  
  private TrainClassifier() { }
  
  public static void trainNaiveBayes(Path dir, Path outputDir, BayesParameters params) throws IOException {
    BayesDriver driver = new BayesDriver();
    driver.runJob(dir, outputDir, params);
  }
  
  public static void trainCNaiveBayes(Path dir, Path outputDir, BayesParameters params) throws IOException {
    CBayesDriver driver = new CBayesDriver();
    driver.runJob(dir, outputDir, params);
  }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Directory on HDFS containing the collapsed, properly formatted files: "
          + "One doc per line, first entry on the line is the label, rest is the evidence")
        .withShortName("i").create();
    
    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "The location of the model on the HDFS").withShortName("o").create();
    
    Option gramSizeOpt = obuilder.withLongName("gramSize").withRequired(true).withArgument(
      abuilder.withName("gramSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "Size of the n-gram. Default Value: 1 ").withShortName("ng").create();
    
    Option alphaOpt = obuilder.withLongName("alpha").withRequired(false).withArgument(
      abuilder.withName("a").withMinimum(1).withMaximum(1).create()).withDescription(
      "Smoothing parameter Default Value: 1.0").withShortName("a").create();
    
    Option typeOpt = obuilder.withLongName("classifierType").withRequired(true).withArgument(
      abuilder.withName("classifierType").withMinimum(1).withMaximum(1).create()).withDescription(
      "Type of classifier: bayes|cbayes. Default: bayes").withShortName("type").create();
    Option dataSourceOpt = obuilder.withLongName("dataSource").withRequired(true).withArgument(
      abuilder.withName("dataSource").withMinimum(1).withMaximum(1).create()).withDescription(
      "Location of model: hdfs|hbase. Default Value: hdfs").withShortName("source").create();
    
    Group group = gbuilder.withName("Options").withOption(gramSizeOpt).withOption(helpOpt).withOption(
      inputDirOpt).withOption(outputOpt).withOption(typeOpt).withOption(dataSourceOpt).withOption(alphaOpt)
        .create();
    try {
      Parser parser = new Parser();
      
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String classifierType = (String) cmdLine.getValue(typeOpt);
      String dataSourceType = (String) cmdLine.getValue(dataSourceOpt);
      
      BayesParameters params = new BayesParameters(Integer.parseInt((String) cmdLine.getValue(gramSizeOpt)));
      
      String alphaI = "1.0";
      if (cmdLine.hasOption(alphaOpt)) {
        alphaI = (String) cmdLine.getValue(alphaOpt);
      }
      
      params.set("alpha_i", alphaI);
      
      if (dataSourceType.equals("hbase")) {
        params.set("dataSource", "hbase");
      } else {
        params.set("dataSource", "hdfs");
      }

      Path inputPath = new Path((String) cmdLine.getValue(inputDirOpt));
      Path outputPath = new Path((String) cmdLine.getValue(outputOpt));
      if (classifierType.equalsIgnoreCase("bayes")) {
        log.info("Training Bayes Classifier");
        trainNaiveBayes(inputPath, outputPath, params);
        
      } else if (classifierType.equalsIgnoreCase("cbayes")) {
        log.info("Training Complementary Bayes Classifier");
        // setup the HDFS and copy the files there, then run the trainer
        trainCNaiveBayes(inputPath, outputPath, params);
      }
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
  }
}
