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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.mahout.utils.strings.StringUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;

public class SequenceFileDumper {

  private static final Logger log = LoggerFactory.getLogger(SequenceFileDumper.class);

  private SequenceFileDumper() {
  }

  public static void main(String[] args) throws IOException, IllegalAccessException, InstantiationException {
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
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt).withOption(substringOpt).withOption(countOpt).create();

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
        JobClient client = new JobClient();
        JobConf conf = new JobConf(Job.class);
        client.setConf(conf);
        FileSystem fs = FileSystem.get(path.toUri(), conf);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

        Writer writer = null;
        if (cmdLine.hasOption(outputOpt)) {
          writer = new FileWriter(cmdLine.getValue(outputOpt).toString());
        } else {
          writer = new OutputStreamWriter(System.out);
        }
        writer.append("Input Path: ").append(String.valueOf(path)).append(StringUtil.LINE_SEP);

        int sub = Integer.MAX_VALUE;
        if (cmdLine.hasOption(substringOpt)) {
          sub = Integer.parseInt(cmdLine.getValue(substringOpt).toString());
        }
        boolean countOnly = cmdLine.hasOption(countOpt);
        long count = 0;
        Writable key = (Writable) reader.getKeyClass().newInstance();
        Writable value = (Writable) reader.getValueClass().newInstance();
        writer.append("Key class: ").append(String.valueOf(reader.getKeyClass())).append(" Value Class: ").append(String.valueOf(value.getClass())).append(StringUtil.LINE_SEP);
        writer.flush();
        if (countOnly == false) {
          while (reader.next(key, value)) {
            writer.append("Key: ").append(String.valueOf(key));
            String str = value.toString();
            writer.append(": Value: ").append(str.length() > sub ? str.substring(0, sub) : str);
            writer.write(StringUtil.LINE_SEP);
            writer.flush();
            count++;
          }
          writer.append("Count: ").append(String.valueOf(count)).append(StringUtil.LINE_SEP);
        } else {
          while (reader.next(key, value)) {
            count++;
          }
          writer.append("Count: ").append(String.valueOf(count)).append(StringUtil.LINE_SEP);
        }
        writer.flush();
        if (cmdLine.hasOption(outputOpt)) {
          writer.close();
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