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

package org.apache.mahout.clustering.lda;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.utils.vectors.VectorHelper;

/**
 * Class to print out the top K words for each topic.
 */
public final class LDAPrintTopics {

  private LDAPrintTopics() { }
  
  // Expands the queue list to have a Queue for topic K
  private static void ensureQueueSize(Collection<Queue<Pair<String,Double>>> queues, int k) {
    for (int i = queues.size(); i <= k; ++i) {
      queues.add(new PriorityQueue<Pair<String,Double>>());
    }
  }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    
    Option dictOpt = obuilder.withLongName("dict").withRequired(true).withArgument(
      abuilder.withName("dict").withMinimum(1).withMaximum(1).create()).withDescription(
      "Dictionary to read in, in the same format as one created by "
          + "org.apache.mahout.utils.vectors.lucene.Driver").withShortName("d").create();
    
    Option outOpt = DefaultOptionCreator.outputOption().create();
    
    Option wordOpt = obuilder.withLongName("words").withRequired(false).withArgument(
      abuilder.withName("words").withMinimum(0).withMaximum(1).withDefault("20").create()).withDescription(
      "Number of words to print").withShortName("w").create();
    Option dictTypeOpt = obuilder.withLongName("dictionaryType").withRequired(false).withArgument(
      abuilder.withName("dictionaryType").withMinimum(1).withMaximum(1).create()).withDescription(
      "The dictionary file type (text|sequencefile)").withShortName("dt").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(dictOpt).withOption(outOpt).withOption(wordOpt)
        .withOption(inputOpt).withOption(dictTypeOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String input = cmdLine.getValue(inputOpt).toString();
      String dictFile = cmdLine.getValue(dictOpt).toString();
      int numWords = 20;
      if (cmdLine.hasOption(wordOpt)) {
        numWords = Integer.parseInt(cmdLine.getValue(wordOpt).toString());
      }
      Configuration config = new Configuration();
      
      String dictionaryType = "text";
      if (cmdLine.hasOption(dictTypeOpt)) {
        dictionaryType = cmdLine.getValue(dictTypeOpt).toString();
      }
      
      List<String> wordList;
      if ("text".equals(dictionaryType)) {
        wordList = Arrays.asList(VectorHelper.loadTermDictionary(new File(dictFile)));
      } else if ("sequencefile".equals(dictionaryType)) {
        wordList = Arrays.asList(VectorHelper.loadTermDictionary(config, dictFile));
      } else {
        throw new IllegalArgumentException("Invalid dictionary format");
      }
      
      List<Queue<Pair<String,Double>>> topWords = topWordsForTopics(input, config, wordList, numWords);

      File output = null;
      if (cmdLine.hasOption(outOpt)) {
        output = new File(cmdLine.getValue(outOpt).toString());
        if (!output.exists() && !output.mkdirs()) {
          throw new IOException("Could not create directory: " + output);
        }
      }
      printTopWords(topWords, output);
    } catch (OptionException e) {
      CommandLineUtil.printHelp(group);
      throw e;
    }
  }
  
  // Adds the word if the queue is below capacity, or the score is high enough
  private static void maybeEnqueue(Queue<Pair<String,Double>> q, String word, double score, int numWordsToPrint) {
    if (q.size() >= numWordsToPrint && score > q.peek().getSecond()) {
      q.poll();
    }
    if (q.size() < numWordsToPrint) {
      q.add(new Pair<String,Double>(word, score));
    }
  }
  
  private static void printTopWords(List<Queue<Pair<String,Double>>> topWords, File outputDir)
    throws IOException {
    for (int i = 0; i < topWords.size(); ++i) {
      Collection<Pair<String,Double>> topK = topWords.get(i);
      Writer out = null;
      boolean printingToSystemOut = false;
      try {
        if (outputDir != null) {
          out = new OutputStreamWriter(new FileOutputStream(new File(outputDir, "topic_" + i)), Charsets.UTF_8);
        } else {
          out = new OutputStreamWriter(System.out, Charsets.UTF_8);
          printingToSystemOut = true;
          out.write("Topic " + i);
          out.write('\n');
          out.write("===========");
          out.write('\n');
        }
        List<Pair<String,Double>> topKasList = Lists.newArrayListWithCapacity(topK.size());
        for (Pair<String,Double> wordWithScore : topK) {
          topKasList.add(wordWithScore);
        }
        Collections.sort(topKasList, new Comparator<Pair<String,Double>>() {
          @Override
          public int compare(Pair<String,Double> pair1, Pair<String,Double> pair2) {
            return pair2.getSecond().compareTo(pair1.getSecond());
          }
        });
        for (Pair<String,Double> wordWithScore : topKasList) {
          out.write(wordWithScore.getFirst() + " [p(" + wordWithScore.getFirst() + "|topic_" + i + ") = "
            + wordWithScore.getSecond());
          out.write('\n');
        }
      } finally {
        if (!printingToSystemOut) {
          Closeables.close(out, false);
        } else {
          out.flush();
        }
      }
    }
  }
  
  private static List<Queue<Pair<String,Double>>> topWordsForTopics(String dir,
                                                                    Configuration job,
                                                                    List<String> wordList,
                                                                    int numWordsToPrint) {
    List<Queue<Pair<String,Double>>> queues = Lists.newArrayList();
    Map<Integer,Double> expSums = Maps.newHashMap();
    for (Pair<IntPairWritable,DoubleWritable> record
        : new SequenceFileDirIterable<IntPairWritable, DoubleWritable>(
            new Path(dir, "part-*"), PathType.GLOB, null, null, true, job)) {
      IntPairWritable key = record.getFirst();
      int topic = key.getFirst();
      int word = key.getSecond();
      ensureQueueSize(queues, topic);
      if (word >= 0 && topic >= 0) {
        double score = record.getSecond().get();
        if (expSums.get(topic) == null) {
          expSums.put(topic, 0.0);
        }
        expSums.put(topic, expSums.get(topic) + Math.exp(score));
        String realWord = wordList.get(word);
        maybeEnqueue(queues.get(topic), realWord, score, numWordsToPrint);
      }
    }
    for (int i = 0; i < queues.size(); i++) {
      Queue<Pair<String,Double>> queue = queues.get(i);
      Queue<Pair<String,Double>> newQueue = new PriorityQueue<Pair<String, Double>>(queue.size());
      double norm = expSums.get(i);
      for (Pair<String,Double> pair : queue) {
        newQueue.add(new Pair<String,Double>(pair.getFirst(), Math.exp(pair.getSecond()) / norm));
      }
      queues.set(i, newQueue);
    }
    return queues;
  }
}
