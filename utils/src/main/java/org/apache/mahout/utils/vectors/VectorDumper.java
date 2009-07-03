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

package org.apache.mahout.utils.vectors;

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
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.mahout.matrix.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;


/**
 * Can read in a {@link org.apache.hadoop.io.SequenceFile} of {@link org.apache.mahout.matrix.Vector}s
 * and dump out the results using {@link org.apache.mahout.matrix.Vector#asFormatString()} to either the console
 * or to a file.
 *
 **/
public final class VectorDumper {

  private static final Logger log = LoggerFactory.getLogger(VectorDumper.class);
  private static final String LINE_SEP = System.getProperty("line.separator");

  private VectorDumper() {
  }

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
            abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).
            withDescription("The Sequence File containing the Vectors").withShortName("s").create();
    Option vectorAsKeyOpt = obuilder.withLongName("useKey").withRequired(false).
            withDescription("If the Key is a vector, then dump that instead").withShortName("u").create();
    Option printKeyOpt = obuilder.withLongName("printKey").withRequired(false).
            withDescription("Print out the key as well, delimited by a tab (or the value if useKey is true)").withShortName("p").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt)
            .withOption(vectorAsKeyOpt).withOption(printKeyOpt).create();

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
        System.out.println("Input Path: " + path);
        JobClient client = new JobClient();
        JobConf conf = new JobConf(Job.class);
        client.setConf(conf);
        FileSystem fs = FileSystem.get(path.toUri(), conf);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
        SequenceFileVectorIterable vectorIterable = new SequenceFileVectorIterable(reader, cmdLine.hasOption(vectorAsKeyOpt));
        Writer writer = null;
        if (cmdLine.hasOption(outputOpt)) {
          writer = new FileWriter(cmdLine.getValue(outputOpt).toString());
        } else {
          writer = new OutputStreamWriter(System.out);
        }
        boolean printKey = cmdLine.hasOption(printKeyOpt);
        SequenceFileVectorIterable.SeqFileIterator iterator =
                (SequenceFileVectorIterable.SeqFileIterator) vectorIterable.iterator();
        int i = 0;
        while (iterator.hasNext()) {
          Vector vector = iterator.next();
          if (printKey){
            writer.write(iterator.key().toString());
            writer.write("\t");
          }
          writer.write(vector.asFormatString());
          writer.write(LINE_SEP);
          i++;
        }
        System.err.println("Dumped " + i + " Vectors");
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
