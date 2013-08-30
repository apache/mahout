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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

import org.apache.commons.io.DirectoryWalker;
import org.apache.commons.io.comparator.CompositeFileComparator;
import org.apache.commons.io.comparator.DirectoryFileComparator;
import org.apache.commons.io.comparator.PathFileComparator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.utils.email.MailOptions;
import org.apache.mahout.utils.email.MailProcessor;
import org.apache.mahout.utils.io.ChunkedWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Converts a directory of gzipped mail archives into SequenceFiles of specified
 * chunkSize. This class is similar to {@link SequenceFilesFromDirectory} except
 * it uses block-compressed {@link org.apache.hadoop.io.SequenceFile}s and parses out the subject and
 * body text of each mail message into a separate key/value pair.
 */
public final class SequenceFilesFromMailArchives extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromMailArchives.class);

  public static final String[] CHUNK_SIZE_OPTION     = {"chunkSize", "chunk"};
  public static final String[] KEY_PREFIX_OPTION     = {"keyPrefix", "prefix"};
  public static final String[] CHARSET_OPTION        = {"charset", "c"};
  public static final String[] SUBJECT_OPTION        = {"subject", "s"};
  public static final String[] TO_OPTION             = {"to", "to"};
  public static final String[] FROM_OPTION           = {"from", "from"};
  public static final String[] REFERENCES_OPTION     = {"references", "refs"};
  public static final String[] BODY_OPTION           = {"body", "b"};
  public static final String[] STRIP_QUOTED_OPTION   = {"stripQuoted", "q"};
  public static final String[] QUOTED_REGEX_OPTION   = {"quotedRegex", "regex"};
  public static final String[] SEPARATOR_OPTION      = {"separator", "sep"};
  public static final String[] BODY_SEPARATOR_OPTION = {"bodySeparator", "bodySep"};
  public static final String BASE_INPUT_PATH         = "baseinputpath";

  private static final int MAX_JOB_SPLIT_LOCATIONS = 1000000;

  public void createSequenceFiles(MailOptions options) throws IOException {
    ChunkedWriter writer = new ChunkedWriter(getConf(), options.getChunkSize(), new Path(options.getOutputDir()));
    MailProcessor processor = new MailProcessor(options, options.getPrefix(), writer);
    try {
      if (options.getInput().isDirectory()) {
        PrefixAdditionDirectoryWalker walker = new PrefixAdditionDirectoryWalker(processor, writer);
        walker.walk(options.getInput());
        log.info("Parsed {} messages from {}", walker.getMessageCount(), options.getInput().getAbsolutePath());
      } else {
        long start = System.currentTimeMillis();
        long cnt = processor.parseMboxLineByLine(options.getInput());
        long finish = System.currentTimeMillis();
        log.info("Parsed {} messages from {} in time: {}", cnt, options.getInput().getAbsolutePath(), finish - start);
      }
    } finally {
      Closeables.close(writer, false);
    }
  }

  private static class PrefixAdditionDirectoryWalker extends DirectoryWalker<Object> {

    @SuppressWarnings("unchecked")
    private static final Comparator<File> FILE_COMPARATOR = new CompositeFileComparator(
        DirectoryFileComparator.DIRECTORY_REVERSE, PathFileComparator.PATH_COMPARATOR);

    private final Deque<MailProcessor> processors = new ArrayDeque<MailProcessor>();
    private final ChunkedWriter writer;
    private final Deque<Long> messageCounts = new ArrayDeque<Long>();

    public PrefixAdditionDirectoryWalker(MailProcessor processor, ChunkedWriter writer) {
      processors.addFirst(processor);
      this.writer = writer;
      messageCounts.addFirst(0L);
    }

    public void walk(File startDirectory) throws IOException {
      super.walk(startDirectory, null);
    }

    public long getMessageCount() {
      return messageCounts.getFirst();
    }

    @Override
    protected void handleDirectoryStart(File current, int depth, Collection<Object> results) throws IOException {
      if (depth > 0) {
        log.info("At {}", current.getAbsolutePath());
        MailProcessor processor = processors.getFirst();
        MailProcessor subDirProcessor = new MailProcessor(processor.getOptions(), processor.getPrefix()
            + File.separator + current.getName(), writer);
        processors.push(subDirProcessor);
        messageCounts.push(0L);
      }
    }

    @Override
    protected File[] filterDirectoryContents(File directory, int depth, File[] files) throws IOException {
      Arrays.sort(files, FILE_COMPARATOR);
      return files;
    }

    @Override
    protected void handleFile(File current, int depth, Collection<Object> results) throws IOException {
      MailProcessor processor = processors.getFirst();
      long currentDirMessageCount = messageCounts.pop();
      try {
        currentDirMessageCount += processor.parseMboxLineByLine(current);
      } catch (IOException e) {
        throw new IllegalStateException("Error processing " + current, e);
      }
      messageCounts.push(currentDirMessageCount);
    }

    @Override
    protected void handleDirectoryEnd(File current, int depth, Collection<Object> results) throws IOException {
      if (depth > 0) {
        final long currentDirMessageCount = messageCounts.pop();
        log.info("Parsed {} messages from directory {}", currentDirMessageCount, current.getAbsolutePath());

        processors.pop();

        // aggregate message counts
        long parentDirMessageCount = messageCounts.pop();
        parentDirMessageCount += currentDirMessageCount;
        messageCounts.push(parentDirMessageCount);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new SequenceFilesFromMailArchives(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.methodOption().create());

    addOption(CHUNK_SIZE_OPTION[0], CHUNK_SIZE_OPTION[1], "The chunkSize in MegaBytes. Defaults to 64", "64");
    addOption(KEY_PREFIX_OPTION[0], KEY_PREFIX_OPTION[1], "The prefix to be prepended to the key", "");
    addOption(CHARSET_OPTION[0], CHARSET_OPTION[1],
      "The name of the character encoding of the input files. Default to UTF-8", "UTF-8");
    addFlag(SUBJECT_OPTION[0], SUBJECT_OPTION[1], "Include the Mail subject as part of the text.  Default is false");
    addFlag(TO_OPTION[0], TO_OPTION[1], "Include the to field in the text.  Default is false");
    addFlag(FROM_OPTION[0], FROM_OPTION[1], "Include the from field in the text.  Default is false");
    addFlag(REFERENCES_OPTION[0], REFERENCES_OPTION[1],
      "Include the references field in the text.  Default is false");
    addFlag(BODY_OPTION[0], BODY_OPTION[1], "Include the body in the output.  Default is false");
    addFlag(STRIP_QUOTED_OPTION[0], STRIP_QUOTED_OPTION[1],
      "Strip (remove) quoted email text in the body.  Default is false");
    addOption(QUOTED_REGEX_OPTION[0], QUOTED_REGEX_OPTION[1],
        "Specify the regex that identifies quoted text.  "
          + "Default is to look for > or | at the beginning of the line.");
    addOption(SEPARATOR_OPTION[0], SEPARATOR_OPTION[1],
        "The separator to use between metadata items (to, from, etc.).  Default is \\n", "\n");
    addOption(BODY_SEPARATOR_OPTION[0], BODY_SEPARATOR_OPTION[1],
        "The separator to use between lines in the body.  Default is \\n.  "
          + "Useful to change if you wish to have the message be on one line", "\n");

    addOption(DefaultOptionCreator.helpOption());
    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    File input = getInputFile();
    String outputDir = getOutputPath().toString();

    int chunkSize = 64;
    if (hasOption(CHUNK_SIZE_OPTION[0])) {
      chunkSize = Integer.parseInt(getOption(CHUNK_SIZE_OPTION[0]));
    }

    String prefix = "";
    if (hasOption(KEY_PREFIX_OPTION[0])) {
      prefix = getOption(KEY_PREFIX_OPTION[0]);
    }

    Charset charset = Charset.forName(getOption(CHARSET_OPTION[0]));
    MailOptions options = new MailOptions();
    options.setInput(input);
    options.setOutputDir(outputDir);
    options.setPrefix(prefix);
    options.setChunkSize(chunkSize);
    options.setCharset(charset);

    List<Pattern> patterns = Lists.newArrayListWithCapacity(5);
    // patternOrder is used downstream so that we can know what order the text
    // is in instead of encoding it in the string, which
    // would require more processing later to remove it pre feature selection.
    Map<String, Integer> patternOrder = Maps.newHashMap();
    int order = 0;
    if (hasOption(FROM_OPTION[0])) {
      patterns.add(MailProcessor.FROM_PREFIX);
      patternOrder.put(MailOptions.FROM, order++);
    }
    if (hasOption(TO_OPTION[0])) {
      patterns.add(MailProcessor.TO_PREFIX);
      patternOrder.put(MailOptions.TO, order++);
    }
    if (hasOption(REFERENCES_OPTION[0])) {
      patterns.add(MailProcessor.REFS_PREFIX);
      patternOrder.put(MailOptions.REFS, order++);
    }
    if (hasOption(SUBJECT_OPTION[0])) {
      patterns.add(MailProcessor.SUBJECT_PREFIX);
      patternOrder.put(MailOptions.SUBJECT, order += 1);
    }
    options.setStripQuotedText(hasOption(STRIP_QUOTED_OPTION[0]));

    options.setPatternsToMatch(patterns.toArray(new Pattern[patterns.size()]));
    options.setPatternOrder(patternOrder);
    options.setIncludeBody(hasOption(BODY_OPTION[0]));

    if (hasOption(SEPARATOR_OPTION[0])) {
      options.setSeparator(getOption(SEPARATOR_OPTION[0]));
    } else {
      options.setSeparator("\n");
    }

    if (hasOption(BODY_SEPARATOR_OPTION[0])) {
      options.setBodySeparator(getOption(BODY_SEPARATOR_OPTION[0]));
    }

    if (hasOption(QUOTED_REGEX_OPTION[0])) {
      options.setQuotedTextPattern(Pattern.compile(getOption(QUOTED_REGEX_OPTION[0])));
    }

    if (getOption(DefaultOptionCreator.METHOD_OPTION,
      DefaultOptionCreator.MAPREDUCE_METHOD).equals(DefaultOptionCreator.SEQUENTIAL_METHOD)) {
      runSequential(options);
    } else {
      runMapReduce(getInputPath(), getOutputPath());
    }

    return 0;
  }

  private int runSequential(MailOptions options)
    throws IOException, InterruptedException, NoSuchMethodException {

    long start = System.currentTimeMillis();
    createSequenceFiles(options);
    long finish = System.currentTimeMillis();
    log.info("Conversion took {}ms", finish - start);

    return 0;
  }

  private int runMapReduce(Path input, Path output) throws IOException, InterruptedException, ClassNotFoundException {

    Job job = prepareJob(input, output, MultipleTextFileInputFormat.class, SequenceFilesFromMailArchivesMapper.class,
      Text.class, Text.class, SequenceFileOutputFormat.class, "SequentialFilesFromMailArchives");

    Configuration jobConfig = job.getConfiguration();

    if (hasOption(KEY_PREFIX_OPTION[0])) {
      jobConfig.set(KEY_PREFIX_OPTION[1], getOption(KEY_PREFIX_OPTION[0]));
    }

    int chunkSize = 0;
    if (hasOption(CHUNK_SIZE_OPTION[0])) {
      chunkSize = Integer.parseInt(getOption(CHUNK_SIZE_OPTION[0]));
      jobConfig.set(CHUNK_SIZE_OPTION[0], String.valueOf(chunkSize));
    }

    Charset charset;
    if (hasOption(CHARSET_OPTION[0])) {
      charset = Charset.forName(getOption(CHARSET_OPTION[0]));
      jobConfig.set(CHARSET_OPTION[0], charset.displayName());
    }

    if (hasOption(FROM_OPTION[0])) {
      jobConfig.set(FROM_OPTION[1], "true");
    }

    if (hasOption(TO_OPTION[0])) {
      jobConfig.set(TO_OPTION[1], "true");
    }

    if (hasOption(REFERENCES_OPTION[0])) {
      jobConfig.set(REFERENCES_OPTION[1], "true");
    }

    if (hasOption(SUBJECT_OPTION[0])) {
      jobConfig.set(SUBJECT_OPTION[1], "true");
    }

    if (hasOption(QUOTED_REGEX_OPTION[0])) {
      jobConfig.set(QUOTED_REGEX_OPTION[1], Pattern.compile(getOption(QUOTED_REGEX_OPTION[0])).toString());
    }

    if (hasOption(SEPARATOR_OPTION[0])) {
      jobConfig.set(SEPARATOR_OPTION[1], getOption(SEPARATOR_OPTION[0]));
    } else {
      jobConfig.set(SEPARATOR_OPTION[1], "\n");
    }

    if (hasOption(BODY_OPTION[0])) {
      jobConfig.set(BODY_OPTION[1], "true");
    } else {
      jobConfig.set(BODY_OPTION[1], "false");
    }

    if (hasOption(BODY_SEPARATOR_OPTION[0])) {
      jobConfig.set(BODY_SEPARATOR_OPTION[1], getOption(BODY_SEPARATOR_OPTION[0]));
    } else {
      jobConfig.set(BODY_SEPARATOR_OPTION[1], "\n");
    }

    FileSystem fs = FileSystem.get(jobConfig);
    FileStatus fsFileStatus = fs.getFileStatus(inputPath);

    jobConfig.set(BASE_INPUT_PATH, inputPath.toString());
    String inputDirList = HadoopUtil.buildDirList(fs, fsFileStatus);
    FileInputFormat.setInputPaths(job, inputDirList);

    long chunkSizeInBytes = chunkSize * 1024 * 1024;
    // need to set this to a multiple of the block size, or no split happens
    FileInputFormat.setMaxInputSplitSize(job, chunkSizeInBytes);

    // set the max split locations, otherwise we get nasty debug stuff
    jobConfig.set("mapreduce.job.max.split.locations", String.valueOf(MAX_JOB_SPLIT_LOCATIONS));

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    return 0;
  }
}
