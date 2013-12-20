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

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Map;

import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.utils.io.ChunkedWriter;

/**
 * Converts a directory of text documents into SequenceFiles of Specified chunkSize. This class takes in a
 * parent directory containing sub folders of text documents and recursively reads the files and creates the
 * {@link org.apache.hadoop.io.SequenceFile}s of docid => content. The docid is set as the relative path of the
 * document from the parent directory prepended with a specified prefix. You can also specify the input encoding
 * of the text files. The content of the output SequenceFiles are encoded as UTF-8 text.
 */
public class SequenceFilesFromDirectory extends AbstractJob {

  private static final String PREFIX_ADDITION_FILTER = PrefixAdditionFilter.class.getName();

  private static final String[] CHUNK_SIZE_OPTION = {"chunkSize", "chunk"};
  public static final String[] FILE_FILTER_CLASS_OPTION = {"fileFilterClass", "filter"};
  private static final String[] CHARSET_OPTION = {"charset", "c"};

  private static final int MAX_JOB_SPLIT_LOCATIONS = 1000000;

  public static final String[] KEY_PREFIX_OPTION = {"keyPrefix", "prefix"};
  public static final String BASE_INPUT_PATH = "baseinputpath";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SequenceFilesFromDirectory(), args);
  }

  /*
  * callback main after processing MapReduce parameters
  */
  @Override
  public int run(String[] args) throws Exception {
    addOptions();
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Map<String, String> options = parseOptions();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }

    if (getOption(DefaultOptionCreator.METHOD_OPTION,
      DefaultOptionCreator.MAPREDUCE_METHOD).equals(DefaultOptionCreator.SEQUENTIAL_METHOD)) {
      runSequential(getConf(), getInputPath(), output, options);
    } else {
      runMapReduce(getInputPath(), output);
    }

    return 0;
  }

  private int runSequential(Configuration conf, Path input, Path output, Map<String, String> options)
    throws IOException, InterruptedException, NoSuchMethodException {
    // Running sequentially
    Charset charset = Charset.forName(getOption(CHARSET_OPTION[0]));
    String keyPrefix = getOption(KEY_PREFIX_OPTION[0]);
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    ChunkedWriter writer = new ChunkedWriter(conf, Integer.parseInt(options.get(CHUNK_SIZE_OPTION[0])), output);

    try {
      SequenceFilesFromDirectoryFilter pathFilter;
      String fileFilterClassName = options.get(FILE_FILTER_CLASS_OPTION[0]);
      if (PrefixAdditionFilter.class.getName().equals(fileFilterClassName)) {
        pathFilter = new PrefixAdditionFilter(conf, keyPrefix, options, writer, charset, fs);
      } else {
        pathFilter = ClassUtils.instantiateAs(fileFilterClassName, SequenceFilesFromDirectoryFilter.class,
          new Class[] {Configuration.class, String.class, Map.class, ChunkedWriter.class, Charset.class, FileSystem.class},
          new Object[] {conf, keyPrefix, options, writer, charset, fs});
      }
      fs.listStatus(input, pathFilter);
    } finally {
      Closeables.close(writer, false);
    }
    return 0;
  }

  private int runMapReduce(Path input, Path output) throws IOException, ClassNotFoundException, InterruptedException {

    int chunkSizeInMB = 64;
    if (hasOption(CHUNK_SIZE_OPTION[0])) {
      chunkSizeInMB = Integer.parseInt(getOption(CHUNK_SIZE_OPTION[0]));
    }

    String keyPrefix = null;
    if (hasOption(KEY_PREFIX_OPTION[0])) {
      keyPrefix = getOption(KEY_PREFIX_OPTION[0]);
    }

    String fileFilterClassName = null;
    if (hasOption(FILE_FILTER_CLASS_OPTION[0])) {
      fileFilterClassName = getOption(FILE_FILTER_CLASS_OPTION[0]);
    }

    PathFilter pathFilter = null;
    // Prefix Addition is presently handled in the Mapper and unlike runsequential()
    // need not be done via a pathFilter
    if (!StringUtils.isBlank(fileFilterClassName) && !PrefixAdditionFilter.class.getName().equals(fileFilterClassName)) {
      try {
        pathFilter = (PathFilter) Class.forName(fileFilterClassName).newInstance();
      } catch (InstantiationException e) {
        throw new IllegalStateException(e);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }

    // Prepare Job for submission.
    Job job = prepareJob(input, output, MultipleTextFileInputFormat.class,
      SequenceFilesFromDirectoryMapper.class, Text.class, Text.class,
      SequenceFileOutputFormat.class, "SequenceFilesFromDirectory");

    Configuration jobConfig = job.getConfiguration();
    jobConfig.set(KEY_PREFIX_OPTION[0], keyPrefix);
    jobConfig.set(FILE_FILTER_CLASS_OPTION[0], fileFilterClassName);

    FileSystem fs = FileSystem.get(jobConfig);
    FileStatus fsFileStatus = fs.getFileStatus(input);

    String inputDirList;
    if (pathFilter != null) {
      inputDirList = HadoopUtil.buildDirList(fs, fsFileStatus, pathFilter);
    } else {
      inputDirList = HadoopUtil.buildDirList(fs, fsFileStatus);
    }

    jobConfig.set(BASE_INPUT_PATH, input.toString());

    long chunkSizeInBytes = chunkSizeInMB * 1024 * 1024;

    // set the max split locations, otherwise we get nasty debug stuff
    jobConfig.set("mapreduce.job.max.split.locations", String.valueOf(MAX_JOB_SPLIT_LOCATIONS));

    FileInputFormat.setInputPaths(job, inputDirList);
    // need to set this to a multiple of the block size, or no split happens
    FileInputFormat.setMaxInputSplitSize(job, chunkSizeInBytes);
    FileOutputFormat.setCompressOutput(job, true);

    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }
    return 0;
  }

  /**
   * Override this method in order to add additional options to the command line of the SequenceFileFromDirectory job.
   * Do not forget to call super() otherwise all standard options (input/output dirs etc) will not be available.
   */
  protected void addOptions() {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(CHUNK_SIZE_OPTION[0], CHUNK_SIZE_OPTION[1], "The chunkSize in MegaBytes. Defaults to 64", "64");
    addOption(FILE_FILTER_CLASS_OPTION[0], FILE_FILTER_CLASS_OPTION[1],
      "The name of the class to use for file parsing. Default: " + PREFIX_ADDITION_FILTER, PREFIX_ADDITION_FILTER);
    addOption(KEY_PREFIX_OPTION[0], KEY_PREFIX_OPTION[1], "The prefix to be prepended to the key", "");
    addOption(CHARSET_OPTION[0], CHARSET_OPTION[1],
      "The name of the character encoding of the input files. Default to UTF-8", "UTF-8");
  }

  /**
   * Override this method in order to parse your additional options from the command line. Do not forget to call
   * super() otherwise standard options (input/output dirs etc) will not be available.
   *
   * @return Map of options
   */
  protected Map<String, String> parseOptions() {
    Map<String, String> options = Maps.newHashMap();
    options.put(CHUNK_SIZE_OPTION[0], getOption(CHUNK_SIZE_OPTION[0]));
    options.put(FILE_FILTER_CLASS_OPTION[0], getOption(FILE_FILTER_CLASS_OPTION[0]));
    options.put(CHARSET_OPTION[0], getOption(CHARSET_OPTION[0]));
    return options;
  }
}
