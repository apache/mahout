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
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.utils.vectors.VectorHelper;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * Class to print out the top K words for each topic.
 */
public final class LDAPrintTopics {

  private LDAPrintTopics() { }  
  
  private static class StringDoublePair implements Comparable<StringDoublePair> {
    private double score;
    private final String word;
    
    StringDoublePair(double score, String word) {
      this.score = score;
      this.word = word;
    }
    
    @Override
    public int compareTo(StringDoublePair other) {
      return Double.compare(score, other.score);
    }
    
    @Override
    public boolean equals(Object o) {
      if (!(o instanceof StringDoublePair)) {
        return false;
      }
      StringDoublePair other = (StringDoublePair) o;
      return score == other.score && word.equals(other.word);
    }
    
    @Override
    public int hashCode() {
      return (int) Double.doubleToLongBits(score) ^ word.hashCode();
    }
    
  }
  
  // Expands the queue list to have a Queue for topic K
  private static void ensureQueueSize(Collection<PriorityQueue<StringDoublePair>> queues, int k) {
    for (int i = queues.size(); i <= k; ++i) {
      queues.add(new PriorityQueue<StringDoublePair>());
    }
  }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "Path to an LDA output (a state)").withShortName("i").create();
    
    Option dictOpt = obuilder.withLongName("dict").withRequired(true).withArgument(
      abuilder.withName("dict").withMinimum(1).withMaximum(1).create()).withDescription(
      "Dictionary to read in, in the same format as one created by "
          + "org.apache.mahout.utils.vectors.lucene.Driver").withShortName("d").create();
    
    Option outOpt = obuilder.withLongName("output").withRequired(false).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "Output directory to write top words").withShortName("o").create();
    
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
      
      List<PriorityQueue<StringDoublePair>> topWords = topWordsForTopics(input, config, wordList, numWords);

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
  private static void maybeEnqueue(Queue<StringDoublePair> q, String word, double score, int numWordsToPrint) {
    if (q.size() >= numWordsToPrint && score > q.peek().score) {
      q.poll();
    }
    if (q.size() < numWordsToPrint) {
      q.add(new StringDoublePair(score, word));
    }
  }
  
  private static void printTopWords(List<PriorityQueue<StringDoublePair>> topWords, File outputDir)
    throws IOException {
    for (int i = 0; i < topWords.size(); ++i) {
      PriorityQueue<StringDoublePair> topK = topWords.get(i);
      PrintWriter out;
      if(outputDir != null) {
        out = new PrintWriter(new File(outputDir, "topic_" + i));
      } else {
        out = new PrintWriter(System.out);
        out.println("Topic " + i);
        out.println("===========");
      }
      List<StringDoublePair> topKasList = new ArrayList<StringDoublePair>(topK.size());
      for(StringDoublePair wordWithScore : topK) {
        topKasList.add(wordWithScore);
      }
      Collections.sort(topKasList, Collections.reverseOrder());
      for(StringDoublePair wordWithScore : topKasList) {
        out.println(wordWithScore.word + " [p(" + wordWithScore.word + "|topic_" + i +") = "
         + wordWithScore.score);
      }
      out.close();
    }
  }
  
  private static List<PriorityQueue<StringDoublePair>> topWordsForTopics(String dir,
                                                      Configuration job,
                                                      List<String> wordList,
                                                      int numWordsToPrint) {
    List<PriorityQueue<StringDoublePair>> queues = new ArrayList<PriorityQueue<StringDoublePair>>();
    Map<Integer,Double> expSums = new HashMap<Integer, Double>();
    for (Pair<IntPairWritable,DoubleWritable> record :
         new SequenceFileDirIterable<IntPairWritable, DoubleWritable>(
             new Path(dir, "part-*"), PathType.GLOB, null, null, true, job)) {
      IntPairWritable key = record.getFirst();
      int topic = key.getFirst();
      int word = key.getSecond();
      ensureQueueSize(queues, topic);
      if (word >= 0 && topic >= 0) {
        double score = record.getSecond().get();
        if(expSums.get(topic) == null) {
          expSums.put(topic, 0d);
        }
        expSums.put(topic, expSums.get(topic) + Math.exp(score));
        String realWord = wordList.get(word);
        maybeEnqueue(queues.get(topic), realWord, score, numWordsToPrint);
      }
    }
    for(int i=0; i<queues.size(); i++) {
      PriorityQueue<StringDoublePair> queue = queues.get(i);
      for(StringDoublePair pair : queue) {
        pair.score = Math.exp(pair.score) / expSums.get(i);
      }
    }
    return queues;
  }
}
