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

package org.apache.mahout.utils;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.OutputStreamWriter;
import java.io.Writer;

public final class SequenceFileDumper {

  private static final Logger log = LoggerFactory.getLogger(SequenceFileDumper.class);

  private SequenceFileDumper() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
            abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).
            withDescription("The Sequence File containing the Clusters").withShortName("s").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option substringOpt = obuilder.withLongName("substring").withRequired(false).withArgument(
            abuilder.withName("substring").withMinimum(1).withMaximum(1).create()).
            withDescription("The number of chars of the asFormatString() to print").withShortName("b").create();
    Option countOpt = obuilder.withLongName("count").withRequired(false).
            withDescription("Report the count only").withShortName("c").create();
    Option numItemsOpt = obuilder.withLongName("n").withRequired(false).withArgument(
            abuilder.withName("numItems").withMinimum(1).withMaximum(1).create()).
            withDescription("Output at most <n> key value pairs").withShortName("n").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt)
            .withOption(substringOpt).withOption(countOpt).withOption(numItemsOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        printHelp(group);
        return;
      }

      if (cmdLine.hasOption(seqOpt)) {
        Path path = new Path(cmdLine.getValue(seqOpt).toString());
        Configuration conf = new Configuration();

        Writer writer;
        if (cmdLine.hasOption(outputOpt)) {
          writer = Files.newWriter(new File(cmdLine.getValue(outputOpt).toString()), Charsets.UTF_8);
        } else {
          writer = new OutputStreamWriter(System.out);
        }
        try {
          writer.append("Input Path: ").append(String.valueOf(path)).append('\n');

          int sub = Integer.MAX_VALUE;
          if (cmdLine.hasOption(substringOpt)) {
            sub = Integer.parseInt(cmdLine.getValue(substringOpt).toString());
          }
          boolean countOnly = cmdLine.hasOption(countOpt);
          SequenceFileIterator<?, ?> iterator = new SequenceFileIterator<Writable, Writable>(path, true, conf);
          writer.append("Key class: ").append(iterator.getKeyClass().toString());
          writer.append(" Value Class: ").append(iterator.getValueClass().toString()).append('\n');
          long count = 0;
          if (countOnly) {
            while (iterator.hasNext()) {
              iterator.next();
              count++;
            }
            writer.append("Count: ").append(String.valueOf(count)).append('\n');
          } else {
            long numItems = Long.MAX_VALUE;
            if (cmdLine.hasOption(numItemsOpt)) {
              numItems = Long.parseLong(cmdLine.getValue(numItemsOpt).toString());
              writer.append("Max Items to dump: ").append(String.valueOf(numItems));
            }
            while (iterator.hasNext() && count < numItems) {
              Pair<?, ?> record = iterator.next();
              writer.append("Key: ").append(record.getFirst().toString());
              String str = record.getSecond().toString();
              writer.append(": Value: ").append(str.length() > sub ? str.substring(0, sub) : str);
              writer.write('\n');
              count++;
            }
            writer.append("Count: ").append(String.valueOf(count)).append('\n');
          }
        } finally {
          Closeables.closeQuietly(writer);
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      printHelp(group);
    }

  }

  private static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
}
