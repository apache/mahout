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
package org.apache.mahout.text;

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
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import org.apache.mahout.utils.email.MailProcessor;
import org.apache.mahout.utils.email.MailOptions;
import org.apache.mahout.utils.io.ChunkedWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Converts a directory of gzipped mail archives into SequenceFiles of specified chunkSize.
 * This class is similar to {@link SequenceFilesFromDirectory} except it uses block-compressed
 * {@link SequenceFile}s and parses out the subject and body text of each mail message into
 * a separate key/value pair.
 */
public final class SequenceFilesFromMailArchives {

  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromMailArchives.class);

  public void createSequenceFiles(MailOptions options) throws IOException {
    ChunkedWriter writer = new ChunkedWriter(new Configuration(), options.getChunkSize(), new Path(options.getOutputDir()));
    MailProcessor processor = new MailProcessor(options, options.getPrefix(), writer);
    try {
      if (options.getInput().isDirectory()) {
        PrefixAdditionFilter filter = new PrefixAdditionFilter(processor, writer);
        options.getInput().listFiles(filter);
        log.info("Parsed {} messages from {}", filter.getMessageCount(), options.getInput().getAbsolutePath());
      } else {
        long start = System.currentTimeMillis();
        long cnt = processor.parseMboxLineByLine(options.getInput());
        long finish = System.currentTimeMillis();
        log.info("Parsed {} messages from {} in time: {}",
                 new Object[] { cnt, options.getInput().getAbsolutePath(), (finish - start) });
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  public class PrefixAdditionFilter implements FileFilter {
    private final MailProcessor processor;
    private final ChunkedWriter writer;
    private long messageCount;

    public PrefixAdditionFilter(MailProcessor processor, ChunkedWriter writer) {
      this.processor = processor;
      this.writer = writer;
      this.messageCount = 0;
    }

    public long getMessageCount() {
      return messageCount;
    }

    @Override
    public boolean accept(File current) {
      if (current.isDirectory()) {
        log.info("At {}", current.getAbsolutePath());
        PrefixAdditionFilter nested = new PrefixAdditionFilter(new MailProcessor(
            processor.getOptions(), processor.getPrefix() + File.separator + current.getName(), writer), writer);
        current.listFiles(nested);
        long dirCount = nested.getMessageCount();
        log.info("Parsed {} messages from directory {}", dirCount, current.getAbsolutePath());
        messageCount += dirCount;
      } else {
        try {
          messageCount += processor.parseMboxLineByLine(current);
        } catch (IOException e) {
          throw new IllegalStateException("Error processing " + current, e);
        }
      }
      return false;
    }
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().create();

    Option outputDirOpt = DefaultOptionCreator.outputOption().create();

    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withArgument(
            abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
            "The chunkSize in MegaBytes. Defaults to 64").withShortName("chunk").create();

    Option keyPrefixOpt = obuilder.withLongName("keyPrefix").withArgument(
            abuilder.withName("keyPrefix").withMinimum(1).withMaximum(1).create()).withDescription(
            "The prefix to be prepended to the key").withShortName("prefix").create();
    Option charsetOpt = obuilder.withLongName("charset").withRequired(true).withArgument(
            abuilder.withName("charset").withMinimum(1).withMaximum(1).create()).withDescription(
            "The name of the character encoding of the input files").withShortName("c").create();
    Option subjectOpt = obuilder.withLongName("subject").withRequired(false).
            withDescription("Include the Mail subject as part of the text.  Default is false").withShortName("s").create();
    Option toOpt = obuilder.withLongName("to").withRequired(false).
            withDescription("Include the to field in the text.  Default is false").withShortName("to").create();
    Option fromOpt = obuilder.withLongName("from").withRequired(false).
            withDescription("Include the from field in the text.  Default is false").withShortName("from").create();
    Option refsOpt = obuilder.withLongName("references").withRequired(false).
            withDescription("Include the references field in the text.  Default is false").withShortName("refs").create();
    Option bodyOpt = obuilder.withLongName("body").withRequired(false).
            withDescription("Include the body in the output.  Default is false").withShortName("b").create();
    Option separatorOpt = obuilder.withLongName("separator").withRequired(false).withArgument(
            abuilder.withName("separator").withMinimum(1).withMaximum(1).create()).
            withDescription("The separator to use between metadata items (to, from, etc.).  Default is \\n").withShortName("sep").create();

    Option bodySeparatorOpt = obuilder.withLongName("bodySeparator").withRequired(false).withArgument(
            abuilder.withName("bodySeparator").withMinimum(1).withMaximum(1).create()).
            withDescription("The separator to use between lines in the body.  Default is \\n.  Useful to change if you wish to have the message be on one line").withShortName("bodySep").create();
    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(keyPrefixOpt).withOption(chunkSizeOpt).withOption(
            charsetOpt).withOption(outputDirOpt).withOption(helpOpt).withOption(inputOpt).withOption(subjectOpt).withOption(toOpt)
            .withOption(fromOpt).withOption(bodyOpt).withOption(refsOpt).withOption(bodySeparatorOpt)
            .withOption(separatorOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      File input = new File((String) cmdLine.getValue(inputOpt));
      String outputDir = (String) cmdLine.getValue(outputDirOpt);

      int chunkSize = 64;
      if (cmdLine.hasOption(chunkSizeOpt)) {
        chunkSize = Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));
      }

      String prefix = "";
      if (cmdLine.hasOption(keyPrefixOpt)) {
        prefix = (String) cmdLine.getValue(keyPrefixOpt);
      }

      Charset charset = Charset.forName((String) cmdLine.getValue(charsetOpt));
      SequenceFilesFromMailArchives dir = new SequenceFilesFromMailArchives();
      MailOptions options = new MailOptions();
      options.setInput(input);
      options.setOutputDir(outputDir);
      options.setPrefix(prefix);
      options.setChunkSize(chunkSize);
      options.setCharset(charset);


      List<Pattern> patterns = new ArrayList<Pattern>(5);
      // patternOrder is used downstream so that we can know what order the text is in instead 
      // of encoding it in the string, which
      // would require more processing later to remove it pre feature selection.
      Map<String, Integer> patternOrder = new HashMap<String, Integer>();
      int order = 0;
      if (cmdLine.hasOption(fromOpt)) {
        patterns.add(MailProcessor.FROM_PREFIX);
        patternOrder.put(MailOptions.FROM, order++);
      }
      if (cmdLine.hasOption(toOpt)) {
        patterns.add(MailProcessor.TO_PREFIX);
        patternOrder.put(MailOptions.TO, order++);
      }
      if (cmdLine.hasOption(refsOpt)) {
        patterns.add(MailProcessor.REFS_PREFIX);
        patternOrder.put(MailOptions.REFS, order++);
      }
      if (cmdLine.hasOption(subjectOpt)) {
        patterns.add(MailProcessor.SUBJECT_PREFIX);
        patternOrder.put(MailOptions.SUBJECT, order++);
      }
      options.setPatternsToMatch(patterns.toArray(new Pattern[patterns.size()]));
      options.setPatternOrder(patternOrder);
      options.setIncludeBody(cmdLine.hasOption(bodyOpt));
      options.setSeparator("\n");
      if (cmdLine.hasOption(separatorOpt)) {
        options.setSeparator(cmdLine.getValue(separatorOpt).toString());
      }
      if (cmdLine.hasOption(bodySeparatorOpt)) {
        options.setBodySeparator(cmdLine.getValue(bodySeparatorOpt).toString());
      }
      long start = System.currentTimeMillis();
      dir.createSequenceFiles(options);
      long finish = System.currentTimeMillis();
      log.info("Conversion took {}ms", finish - start);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

}
