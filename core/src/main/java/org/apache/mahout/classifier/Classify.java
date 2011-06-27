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

package org.apache.mahout.classifier;

import java.io.File;
import java.nio.charset.Charset;
import java.util.List;

import com.google.common.io.Files;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.bayes.Algorithm;
import org.apache.mahout.classifier.bayes.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.CBayesAlgorithm;
import org.apache.mahout.classifier.bayes.Datastore;
import org.apache.mahout.classifier.bayes.ClassifierContext;
import org.apache.mahout.classifier.bayes.InMemoryBayesDatastore;
import org.apache.mahout.common.nlp.NGrams;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs the Bayes classifier using the given model location on HDFS
 * 
 */
public final class Classify {
  
  private static final Logger log = LoggerFactory.getLogger(Classify.class);
  
  private Classify() { }
  
  public static void main(String[] args) throws Exception {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option pathOpt = obuilder.withLongName("path").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription(
      "The local file system path").withShortName("m").create();
    
    Option classifyOpt = obuilder.withLongName("classify").withRequired(true).withArgument(
      abuilder.withName("classify").withMinimum(1).withMaximum(1).create()).withDescription(
      "The doc to classify").withShortName("").create();
    
    Option encodingOpt = obuilder.withLongName("encoding").withRequired(true).withArgument(
      abuilder.withName("encoding").withMinimum(1).withMaximum(1).create()).withDescription(
      "The file encoding.  Default: UTF-8").withShortName("e").create();
    
    Option analyzerOpt = obuilder.withLongName("analyzer").withRequired(true).withArgument(
      abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Analyzer to use").withShortName("a").create();
    
    Option defaultCatOpt = obuilder.withLongName("defaultCat").withRequired(true).withArgument(
      abuilder.withName("defaultCat").withMinimum(1).withMaximum(1).create()).withDescription(
      "The default category").withShortName("d").create();
    
    Option gramSizeOpt = obuilder.withLongName("gramSize").withRequired(true).withArgument(
      abuilder.withName("gramSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "Size of the n-gram").withShortName("ng").create();
    
    Option typeOpt = obuilder.withLongName("classifierType").withRequired(true).withArgument(
      abuilder.withName("classifierType").withMinimum(1).withMaximum(1).create()).withDescription(
      "Type of classifier").withShortName("type").create();
    
    Option dataSourceOpt = obuilder.withLongName("dataSource").withRequired(true).withArgument(
      abuilder.withName("dataSource").withMinimum(1).withMaximum(1).create()).withDescription(
      "Location of model: hdfs").withShortName("source").create();
    
    Group options = gbuilder.withName("Options").withOption(pathOpt).withOption(classifyOpt).withOption(
      encodingOpt).withOption(analyzerOpt).withOption(defaultCatOpt).withOption(gramSizeOpt).withOption(
      typeOpt).withOption(dataSourceOpt).create();
    
    Parser parser = new Parser();
    parser.setGroup(options);
    CommandLine cmdLine = parser.parse(args);
    
    int gramSize = 1;
    if (cmdLine.hasOption(gramSizeOpt)) {
      gramSize = Integer.parseInt((String) cmdLine.getValue(gramSizeOpt));
      
    }
    
    BayesParameters params = new BayesParameters();
    params.setGramSize(gramSize);
    String modelBasePath = (String) cmdLine.getValue(pathOpt);
    params.setBasePath(modelBasePath);

    log.info("Loading model from: {}", params.print());
    
    Algorithm algorithm;
    Datastore datastore;
    
    String classifierType = (String) cmdLine.getValue(typeOpt);
    
    String dataSource = (String) cmdLine.getValue(dataSourceOpt);
    if ("hdfs".equals(dataSource)) {
      if ("bayes".equalsIgnoreCase(classifierType)) {
        log.info("Using Bayes Classifier");
        algorithm = new BayesAlgorithm();
        datastore = new InMemoryBayesDatastore(params);
      } else if ("cbayes".equalsIgnoreCase(classifierType)) {
        log.info("Using Complementary Bayes Classifier");
        algorithm = new CBayesAlgorithm();
        datastore = new InMemoryBayesDatastore(params);
      } else {
        throw new IllegalArgumentException("Unrecognized classifier type: " + classifierType);
      }
      
    } else {
      throw new IllegalArgumentException("Unrecognized dataSource type: " + dataSource);
    }
    ClassifierContext classifier = new ClassifierContext(algorithm, datastore);
    classifier.initialize();
    String defaultCat = "unknown";
    if (cmdLine.hasOption(defaultCatOpt)) {
      defaultCat = (String) cmdLine.getValue(defaultCatOpt);
    }
    File docPath = new File((String) cmdLine.getValue(classifyOpt));
    String encoding = "UTF-8";
    if (cmdLine.hasOption(encodingOpt)) {
      encoding = (String) cmdLine.getValue(encodingOpt);
    }
    Analyzer analyzer = null;
    if (cmdLine.hasOption(analyzerOpt)) {
      String className = (String) cmdLine.getValue(analyzerOpt);
      analyzer = Class.forName(className).asSubclass(Analyzer.class).newInstance();
    }
    if (analyzer == null) {
      analyzer = new StandardAnalyzer(Version.LUCENE_31);
    }
    
    log.info("Converting input document to proper format");

    String[] document =
        BayesFileFormatter.readerToDocument(analyzer,Files.newReader(docPath, Charset.forName(encoding)));
    StringBuilder line = new StringBuilder();
    for (String token : document) {
      line.append(token).append(' ');
    }
    
    List<String> doc = new NGrams(line.toString(), gramSize).generateNGramsWithoutLabel();
    
    log.info("Done converting");
    log.info("Classifying document: {}", docPath);
    ClassifierResult category = classifier.classifyDocument(doc.toArray(new String[doc.size()]), defaultCat);
    log.info("Category for {} is {}", docPath, category);
    
  }
}
