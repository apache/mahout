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

package org.apache.mahout.clustering.streaming.tools;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Iterator;

import com.google.common.base.Charsets;
import com.google.common.collect.Iterables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;

public class ResplitSequenceFiles {

  private String inputFile;
  private String outputFileBase;
  private int numSplits;

  private Configuration conf;
  private FileSystem fs;

  private ResplitSequenceFiles() {}

  private void writeSplit(Iterator<Pair<Writable, Writable>> inputIterator,
                          int numSplit, int numEntriesPerSplit) throws IOException {
    SequenceFile.Writer splitWriter = null;
    for (int j = 0; j < numEntriesPerSplit; ++j) {
      Pair<Writable, Writable> item = inputIterator.next();
      if (splitWriter == null) {
        splitWriter = SequenceFile.createWriter(fs, conf,
            new Path(outputFileBase + "-" + numSplit), item.getFirst().getClass(), item.getSecond().getClass());
      }
      splitWriter.append(item.getFirst(), item.getSecond());
    }
    if (splitWriter != null) {
      splitWriter.close();
    }
  }

  private void run(PrintWriter printWriter) throws IOException {
    conf = new Configuration();
    SequenceFileDirIterable<Writable, Writable> inputIterable = new
        SequenceFileDirIterable<Writable, Writable>(new Path(inputFile), PathType.LIST, conf);
    fs = FileSystem.get(conf);

    int numEntries = Iterables.size(inputIterable);
    int numEntriesPerSplit = numEntries / numSplits;
    int numEntriesLastSplit = numEntriesPerSplit + numEntries - numEntriesPerSplit * numSplits;
    Iterator<Pair<Writable, Writable>> inputIterator = inputIterable.iterator();

    printWriter.printf("Writing %d splits\n", numSplits);
    for (int i = 0; i < numSplits - 1; ++i) {
      printWriter.printf("Writing split %d\n", i);
      writeSplit(inputIterator, i, numEntriesPerSplit);
    }
    printWriter.printf("Writing split %d\n", numSplits - 1);
    writeSplit(inputIterator, numSplits - 1, numEntriesLastSplit);
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("what the base folder for sequence files is (they all must have the same key/value type")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the base name of the file split that the files will be split it; the i'th split has the "
            + "suffix -i")
        .create();

    Option numSplitsOption = builder.withLongName("numSplits")
        .withShortName("ns")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("numSplits").withMaximum(1).create())
        .withDescription("how many splits to use for the given files")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(numSplitsOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));
    CommandLine cmdLine = parser.parseAndHelp(args);

    if (cmdLine == null) {
      return false;
    }

    inputFile = (String) cmdLine.getValue(inputFileOption);
    outputFileBase = (String) cmdLine.getValue(outputFileOption);
    numSplits = Integer.parseInt((String) cmdLine.getValue(numSplitsOption));
    return true;
  }

  public static void main(String[] args) throws IOException {
    ResplitSequenceFiles runner = new ResplitSequenceFiles();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }
}
