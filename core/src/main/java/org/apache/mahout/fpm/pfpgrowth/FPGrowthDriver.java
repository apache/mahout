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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextStatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.convertors.SequenceFileOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.StringOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FPGrowthDriver {
  
  private static final Logger log = LoggerFactory.getLogger(FPGrowthDriver.class);
  
  private FPGrowthDriver() { }
  
  /**
   * Run TopK FPGrowth given the input file,
   */
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Directory on HDFS containing the transaction files").withShortName("i").create();
    
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    // minSupport(3), maxHeapSize(50), numGroups(1000)
    Option minSupportOpt = obuilder.withLongName("minSupport").withArgument(
      abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Minimum Support. Default Value: 3").withShortName("s").create();
    
    Option maxHeapSizeOpt = obuilder.withLongName("maxHeapSize").withArgument(
      abuilder.withName("maxHeapSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Maximum Heap Size k, to denote the requirement to mine top K items. Default value: 50")
        .withShortName("k").create();
    
    Option numGroupsOpt = obuilder.withLongName("numGroups").withArgument(
      abuilder.withName("numGroups").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Number of groups the features should be divided in the map-reduce version."
          + " Doesn't work in sequential version Default Value:1000").withShortName("g").create();
    
    Option recordSplitterOpt = obuilder.withLongName("splitterPattern").withArgument(
      abuilder.withName("splitterPattern").withMinimum(1).withMaximum(1).create()).withDescription(
      "Regular Expression pattern used to split given string transaction into itemsets."
          + " Default value splits comma separated itemsets.  Default Value:"
          + " \"[ ,\\t]*[,|\\t][ ,\\t]*\" ").withShortName("regex").create();
    
    Option treeCacheOpt = obuilder.withLongName("numTreeCacheEntries").withArgument(
      abuilder.withName("numTreeCacheEntries").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) Number of entries in the tree cache to prevent duplicate tree building. "
          + "(Warning) a first level conditional FP-Tree might consume a lot of memory, "
          + "so keep this value small, but big enough to prevent duplicate tree building. "
          + "Default Value:5 Recommended Values: [5-10]").withShortName("tc").create();
    
    Option methodOpt = obuilder.withLongName("method").withRequired(true).withArgument(
      abuilder.withName("method").withMinimum(1).withMaximum(1).create()).withDescription(
      "Method of processing: sequential|mapreduce").withShortName("method").create();
    Option encodingOpt = obuilder.withLongName("encoding").withArgument(
      abuilder.withName("encoding").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) The file encoding.  Default value: UTF-8").withShortName("e").create();
    
    Group group = gbuilder.withName("Options").withOption(minSupportOpt).withOption(inputDirOpt).withOption(
      outputOpt).withOption(maxHeapSizeOpt).withOption(numGroupsOpt).withOption(methodOpt).withOption(
      encodingOpt).withOption(helpOpt).withOption(treeCacheOpt).withOption(recordSplitterOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      Parameters params = new Parameters();
      
      if (cmdLine.hasOption(minSupportOpt)) {
        String minSupportString = (String) cmdLine.getValue(minSupportOpt);
        params.set("minSupport", minSupportString);
      }
      if (cmdLine.hasOption(maxHeapSizeOpt)) {
        String maxHeapSizeString = (String) cmdLine.getValue(maxHeapSizeOpt);
        params.set("maxHeapSize", maxHeapSizeString);
      }
      if (cmdLine.hasOption(numGroupsOpt)) {
        String numGroupsString = (String) cmdLine.getValue(numGroupsOpt);
        params.set("numGroups", numGroupsString);
      }
      
      if (cmdLine.hasOption(treeCacheOpt)) {
        String numTreeCacheString = (String) cmdLine.getValue(treeCacheOpt);
        params.set("treeCacheSize", numTreeCacheString);
      }
      
      if (cmdLine.hasOption(recordSplitterOpt)) {
        String patternString = (String) cmdLine.getValue(recordSplitterOpt);
        params.set("splitPattern", patternString);
      }
      
      String encoding = "UTF-8";
      if (cmdLine.hasOption(encodingOpt)) {
        encoding = (String) cmdLine.getValue(encodingOpt);
      }
      params.set("encoding", encoding);
      Path inputDir = new Path(cmdLine.getValue(inputDirOpt).toString());
      Path outputDir = new Path(cmdLine.getValue(outputOpt).toString());
      
      params.set("input", inputDir.toString());
      params.set("output", outputDir.toString());
      
      String classificationMethod = (String) cmdLine.getValue(methodOpt);
      if ("sequential".equalsIgnoreCase(classificationMethod)) {
        runFPGrowth(params);
      } else if ("mapreduce".equalsIgnoreCase(classificationMethod)) {
        Configuration conf = new Configuration();
        HadoopUtil.delete(conf, outputDir);
        PFPGrowth.runPFPGrowth(params);
      }
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
    }
  }
  
  private static void runFPGrowth(Parameters params) throws IOException {
    log.info("Starting Sequential FPGrowth");
    int maxHeapSize = Integer.valueOf(params.get("maxHeapSize", "50"));
    int minSupport = Integer.valueOf(params.get("minSupport", "3"));
    
    String output = params.get("output", "output.txt");
    
    Path path = new Path(output);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    
    Charset encoding = Charset.forName(params.get("encoding"));
    String input = params.get("input");
    
    String pattern = params.get("splitPattern", PFPGrowth.SPLITTER.toString());
    
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, TopKStringPatterns.class);
    
    FPGrowth<String> fp = new FPGrowth<String>();
    Collection<String> features = new HashSet<String>();

    try {
      fp.generateTopKFrequentPatterns(
          new StringRecordIterator(new FileLineIterable(new File(input), encoding, false), pattern),
          fp.generateFList(
              new StringRecordIterator(new FileLineIterable(new File(input), encoding, false), pattern),
              minSupport),
          minSupport,
          maxHeapSize,
          features,
          new StringOutputConverter(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer)),
          new ContextStatusUpdater(null));
    } finally {
      Closeables.closeQuietly(writer);
    }
    
    List<Pair<String,TopKStringPatterns>> frequentPatterns = FPGrowth.readFrequentPattern(conf, path);
    for (Pair<String,TopKStringPatterns> entry : frequentPatterns) {
      log.info("Dumping Patterns for Feature: {} \n{}", entry.getFirst(), entry.getSecond());
    }
  }
}
