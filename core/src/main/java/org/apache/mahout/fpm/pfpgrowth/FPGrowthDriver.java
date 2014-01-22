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

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextStatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.convertors.SequenceFileOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.StringOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FPGrowthDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(FPGrowthDriver.class);

  private FPGrowthDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new FPGrowthDriver(), args);
  }

  /**
   * Run TopK FPGrowth given the input file,
   */
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();

    addOption("minSupport", "s", "(Optional) The minimum number of times a co-occurrence must be present."
              + " Default Value: 3", "3");
    addOption("maxHeapSize", "k", "(Optional) Maximum Heap Size k, to denote the requirement to mine top K items."
              + " Default value: 50", "50");
    addOption("numGroups", "g", "(Optional) Number of groups the features should be divided in the map-reduce version."
              + " Doesn't work in sequential version Default Value:" + PFPGrowth.NUM_GROUPS_DEFAULT,
              Integer.toString(PFPGrowth.NUM_GROUPS_DEFAULT));
    addOption("splitterPattern", "regex", "Regular Expression pattern used to split given string transaction into"
            + " itemsets. Default value splits comma separated itemsets.  Default Value:"
            + " \"[ ,\\t]*[,|\\t][ ,\\t]*\" ", "[ ,\t]*[,|\t][ ,\t]*");
    addOption("numTreeCacheEntries", "tc", "(Optional) Number of entries in the tree cache to prevent duplicate"
            + " tree building. (Warning) a first level conditional FP-Tree might consume a lot of memory, "
            + "so keep this value small, but big enough to prevent duplicate tree building. "
            + "Default Value:5 Recommended Values: [5-10]", "5");
    addOption("method", "method", "Method of processing: sequential|mapreduce", "sequential");
    addOption("encoding", "e", "(Optional) The file encoding.  Default value: UTF-8", "UTF-8");
    addFlag("useFPG2", "2", "Use an alternate FPG implementation");

    if (parseArguments(args) == null) {
      return -1;
    }

    Parameters params = new Parameters();

    if (hasOption("minSupport")) {
      String minSupportString = getOption("minSupport");
      params.set("minSupport", minSupportString);
    }
    if (hasOption("maxHeapSize")) {
      String maxHeapSizeString = getOption("maxHeapSize");
      params.set("maxHeapSize", maxHeapSizeString);
    }
    if (hasOption("numGroups")) {
      String numGroupsString = getOption("numGroups");
      params.set("numGroups", numGroupsString);
    }

    if (hasOption("numTreeCacheEntries")) {
      String numTreeCacheString = getOption("numTreeCacheEntries");
      params.set("treeCacheSize", numTreeCacheString);
    }

    if (hasOption("splitterPattern")) {
      String patternString = getOption("splitterPattern");
      params.set("splitPattern", patternString);
    }

    String encoding = "UTF-8";
    if (hasOption("encoding")) {
      encoding = getOption("encoding");
    }
    params.set("encoding", encoding);

    if (hasOption("useFPG2")) {
      params.set(PFPGrowth.USE_FPG2, "true");
    }

    Path inputDir = getInputPath();
    Path outputDir = getOutputPath();

    params.set("input", inputDir.toString());
    params.set("output", outputDir.toString());

    String classificationMethod = getOption("method");
    if ("sequential".equalsIgnoreCase(classificationMethod)) {
      runFPGrowth(params);
    } else if ("mapreduce".equalsIgnoreCase(classificationMethod)) {
      Configuration conf = new Configuration();
      HadoopUtil.delete(conf, outputDir);
      PFPGrowth.runPFPGrowth(params);
    }

    return 0;
  }

  private static void runFPGrowth(Parameters params) throws IOException {
    log.info("Starting Sequential FPGrowth");
    int maxHeapSize = Integer.valueOf(params.get("maxHeapSize", "50"));
    int minSupport = Integer.valueOf(params.get("minSupport", "3"));

    Path output = new Path(params.get("output", "output.txt"));
    Path input = new Path(params.get("input"));

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(output.toUri(), conf);

    Charset encoding = Charset.forName(params.get("encoding"));

    String pattern = params.get("splitPattern", PFPGrowth.SPLITTER.toString());

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, output, Text.class, TopKStringPatterns.class);

    FSDataInputStream inputStream = null;
    FSDataInputStream inputStreamAgain = null;

    Collection<String> features = Sets.newHashSet();

    if ("true".equals(params.get(PFPGrowth.USE_FPG2))) {
      org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj<String> fp 
        = new org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj<String>();

      try {
        inputStream = fs.open(input);
        inputStreamAgain = fs.open(input);
        fp.generateTopKFrequentPatterns(
                new StringRecordIterator(new FileLineIterable(inputStream, encoding, false), pattern),
                fp.generateFList(
                        new StringRecordIterator(new FileLineIterable(inputStreamAgain, encoding, false), pattern),
                        minSupport),
                minSupport,
                maxHeapSize,
                features,
                new StringOutputConverter(new SequenceFileOutputCollector<Text, TopKStringPatterns>(writer))
        );
      } finally {
        Closeables.close(writer, false);
        Closeables.close(inputStream, true);
        Closeables.close(inputStreamAgain, true);
      }
    } else {
      FPGrowth<String> fp = new FPGrowth<String>();


      inputStream = fs.open(input);
      inputStreamAgain = fs.open(input);
      try {
        fp.generateTopKFrequentPatterns(
                new StringRecordIterator(new FileLineIterable(inputStream, encoding, false), pattern),
                fp.generateFList(
                        new StringRecordIterator(new FileLineIterable(inputStreamAgain, encoding, false), pattern),
                        minSupport),
                minSupport,
                maxHeapSize,
                features,
                new StringOutputConverter(new SequenceFileOutputCollector<Text, TopKStringPatterns>(writer)),
                new ContextStatusUpdater(null));
      } finally {
        Closeables.close(writer, false);
        Closeables.close(inputStream, true);
        Closeables.close(inputStreamAgain, true);
      }
    }

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowth.readFrequentPattern(conf, output);
    for (Pair<String, TopKStringPatterns> entry : frequentPatterns) {
      log.info("Dumping Patterns for Feature: {} \n{}", entry.getFirst(), entry.getSecond());
    }
  }
}
