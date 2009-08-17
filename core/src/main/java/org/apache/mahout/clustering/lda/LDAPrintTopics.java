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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.utils.CommandLineUtil;

/**
 * Class to print out the top K words for each topic.
 */
public class LDAPrintTopics {
  private LDAPrintTopics() {
  }

  private static class StringDoublePair implements Comparable<StringDoublePair> {
    StringDoublePair(double score, String word) {
      this.score = score;
      this.word = word;
    }
    
    public int compareTo(StringDoublePair other) {
      return Double.compare(score,other.score);
    }

    double score;
    String word;
  }

  public static List<List<String>> topWordsForTopics(String dir, Configuration job,
      List<String> wordList, int numWordsToPrint) throws IOException {
    FileSystem fs = new Path(dir).getFileSystem(job);

    List<PriorityQueue<StringDoublePair>> queues = new ArrayList<PriorityQueue<StringDoublePair>>();

    IntPairWritable key = new IntPairWritable();
    DoubleWritable value = new DoubleWritable();
    for (FileStatus status : fs.globStatus(new Path(dir, "*"))) { 
      Path path = status.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
      while (reader.next(key, value)) {
        int topic = key.getX();
        int word = key.getY();

        ensureQueueSize(queues,topic);
        if (word >= 0 && topic >= 0) {
          double score = value.get();
          String realWord = wordList.get(word);
          maybeEnqueue(queues.get(topic), realWord, score, numWordsToPrint);
        }
      }
      reader.close();
    }

    List<List<String>> result = new ArrayList<List<String>>();
    for (int i = 0; i < queues.size(); ++i) {
      result.add(i,new LinkedList<String>());
      for (StringDoublePair sdp: queues.get(i)) {
        result.get(i).add(0,sdp.word); // prepend
      }
    }

    return result;
  }

  // Expands the queue list to have a Queue for topic K
  private static void ensureQueueSize(List<PriorityQueue<StringDoublePair>> queues, int k) {
    for (int i = queues.size(); i <= k; ++i) {
      queues.add(new PriorityQueue<StringDoublePair>());
    }
  }

  // Adds the word if the queue is below capacity, or the score is high enough
  private static void maybeEnqueue(PriorityQueue<StringDoublePair> q, String word, 
      double score, int numWordsToPrint) {
    if (q.size() >= numWordsToPrint && score > q.peek().score) {
      q.poll();
    }
    if (q.size() < numWordsToPrint) {
      q.add(new StringDoublePair(score,word));
    } 
  }

  // Reads dictionary in created by the vector Driver in util
  private static List<String> readDictionary(File path) throws IOException {
    BufferedReader rdr = new BufferedReader(new FileReader(path));

    List<String> result = new ArrayList<String>();

    // skip 2 lines
    rdr.readLine();
    rdr.readLine();
    String line = null;
    while ( (line = rdr.readLine()) != null) {
      String[] parts = line.split("\t");
      String word = parts[0];
      int index = Integer.parseInt(parts[2]);
      assert index == result.size();
      result.add(word);
    }
    rdr.close();

    return result;
  }

  public static void main(String[] args) {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "Path to an LDA output (a state)").withShortName("i").create();

    Option dictOpt = obuilder.withLongName("dict").withRequired(true).withArgument(
        abuilder.withName("dict").withMinimum(1).withMaximum(1).create()).withDescription(
        "Dictionary to read in, created by utils.vector.Driver").withShortName("d").create();

    Option outOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "Output directory to write top words").withShortName("o").create();

    Option wordOpt = obuilder.withLongName("words").withRequired(true).withArgument(
        abuilder.withName("words").withMinimum(0).withMaximum(1).withDefault("20").create()).withDescription(
        "Number of words to print").withShortName("w").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(dictOpt).withOption(outOpt).withOption(
        wordOpt).withOption(inputOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      String input   = cmdLine.getValue(inputOpt).toString();
      File output = new File(cmdLine.getValue(outOpt).toString());
      File dict   = new File(cmdLine.getValue(dictOpt).toString());
      int numWords = 20;
      if (cmdLine.hasOption(wordOpt)) {
        numWords = Integer.parseInt(cmdLine.getValue(wordOpt).toString());
      }

      List<String> wordList = readDictionary(dict);

      Configuration config = new Configuration();
      List<List<String>> topWords = topWordsForTopics(input, config, wordList, numWords);

      if(!output.exists()) {
        if (!output.mkdirs()) {
          throw new IOException("Could not create directory: " + output);
        }
      }

      for (int i = 0; i < topWords.size(); ++i) {
        List<String> topK = topWords.get(i);
        File out = new File(output,"topic-"+i);
        PrintWriter writer = new PrintWriter(new FileWriter(out));
        writer.println("Topic " + i);
        writer.println("===========");
        for (String word: topK) {
          writer.println(word);
        }
        writer.close();
      }

    } catch (OptionException e) {
      System.err.println("Exception: " + e);
      CommandLineUtil.printHelp(group);
    } catch (IOException e) {
      System.err.println("Exception:" + e);
      e.printStackTrace();
    }
  }

}
