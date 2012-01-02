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

package org.apache.mahout.common;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.vectorizer.DefaultAnalyzer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>Superclass of many Mahout Hadoop "jobs". A job drives configuration and launch of one or
 * more maps and reduces in order to accomplish some task.</p>
 *
 * <p>Command line arguments available to all subclasses are:</p>
 *
 * <ul>
 *  <li>--tempDir (path): Specifies a directory where the job may place temp files
 *   (default "temp")</li>
 *  <li>--help: Show help message</li>
 * </ul>
 *
 * <p>In addition, note some key command line parameters that are parsed by Hadoop, which jobs
 * may need to set:</p>
 *
 * <ul>
 *  <li>-Dmapred.job.name=(name): Sets the Hadoop task names. It will be suffixed by
 *    the mapper and reducer class names</li>
 *  <li>-Dmapred.output.compress={true,false}: Compress final output (default true)</li>
 *  <li>-Dmapred.input.dir=(path): input file, or directory containing input files (required)</li>
 *  <li>-Dmapred.output.dir=(path): path to write output files (required)</li>
 * </ul>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public abstract class AbstractJob extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(AbstractJob.class);

  /** option used to specify the input path */
  private Option inputOption;

  /** option used to specify the output path */
  private Option outputOption;

  /** input path, populated by {@link #parseArguments(String[])} */
  private Path inputPath;

  /** output path, populated by {@link #parseArguments(String[]) */
  private Path outputPath;

  /** temp path, populated by {@link #parseArguments(String[]) */
  private Path tempPath;

  private Map<String, String> argMap;

  /** internal list of options that have been added */
  private final List<Option> options;
  protected Group group;

  protected AbstractJob() {
    options = new LinkedList<Option>();
  }

  /** Returns the input path established by a call to {@link #parseArguments(String[])}.
   *  The source of the path may be an input option added using {@link #addInputOption()}
   *  or it may be the value of the {@code mapred.input.dir} configuration
   *  property. 
   */
  protected Path getInputPath() {
    return inputPath;
  }

  /** Returns the output path established by a call to {@link #parseArguments(String[])}.
   *  The source of the path may be an output option added using {@link #addOutputOption()}
   *  or it may be the value of the {@code mapred.input.dir} configuration
   *  property. 
   */
  protected Path getOutputPath() {
    return outputPath;
  }

  protected Path getOutputPath(String path) {
    return new Path(outputPath, path);
  }


  protected Path getTempPath() {
    return tempPath;
  }

  protected Path getTempPath(String directory) {
    return new Path(tempPath, directory);
  }

  protected Path getCombinedTempPath(String directory1, String directory2) {
    return new Path(new Path(tempPath, directory1) + "," + new Path(tempPath, directory2));
  }

  /** Add an option with no argument whose presence can be checked for using
   *  {@code containsKey} method on the map returned by {@link #parseArguments(String[])};
   */
  protected void addFlag(String name, String shortName, String description) {
    options.add(buildOption(name, shortName, description, false, false, null));
  }

  /** Add an option to the the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called. This options has an argument
   *  with null as its default value.
   */
  protected void addOption(String name, String shortName, String description) {
    options.add(buildOption(name, shortName, description, true, false, null));
  }

  /** Add an option to the the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called.
   *
   * @param required if true the {@link #parseArguments(String[])} will throw
   *    fail with an error and usage message if this option is not specified
   *    on the command line.
   */
  protected void addOption(String name, String shortName, String description, boolean required) {
    options.add(buildOption(name, shortName, description, true, required, null));
  }

  /** Add an option to the the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called. If this option is not 
   *  specified on the command line the default value will be 
   *  used.
   *
   * @param defaultValue the default argument value if this argument is not
   *   found on the command-line. null is allowed.
   */
  protected void addOption(String name, String shortName, String description, String defaultValue) {
    options.add(buildOption(name, shortName, description, true, false, defaultValue));
  }

  /** Add an arbitrary option to the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called. If this option has no
   *  argument, use {@code containsKey} on the map returned by
   *  {@code parseArguments} to check for its presence. Otherwise, the
   *  string value of the option will be placed in the map using a key
   *  equal to this options long name preceded by '--'.
   * @return the option added.
   */
  protected Option addOption(Option option) {
    options.add(option);
    return option;
  }

  /** Add the default input directory option, '-i' which takes a directory
   *  name as an argument. When {@link #parseArguments(String[])} is 
   *  called, the inputPath will be set based upon the value for this option.
   *  If this method is called, the input is required.
   */
  protected void addInputOption() {
    this.inputOption = addOption(DefaultOptionCreator.inputOption().create());
  }

  /** Add the default output directory option, '-o' which takes a directory
   *  name as an argument. When {@link #parseArguments(String[])} is 
   *  called, the outputPath will be set based upon the value for this option.
   *  If this method is called, the output is required. 
   */
  protected void addOutputOption() {
    this.outputOption = addOption(DefaultOptionCreator.outputOption().create());
  }

  /** Build an option with the given parameters. Name and description are
   *  required.
   * 
   * @param name the long name of the option prefixed with '--' on the command-line
   * @param shortName the short name of the option, prefixed with '-' on the command-line
   * @param description description of the option displayed in help method
   * @param hasArg true if the option has an argument.
   * @param required true if the option is required.
   * @param defaultValue default argument value, can be null.
   * @return the option.
   */
  protected static Option buildOption(String name,
                                    String shortName,
                                    String description,
                                    boolean hasArg,
                                    boolean required,
                                    String defaultValue) {

    DefaultOptionBuilder optBuilder = new DefaultOptionBuilder().withLongName(name).withDescription(description)
        .withRequired(required);

    if (shortName != null) {
      optBuilder.withShortName(shortName);
    }

    if (hasArg) {
      ArgumentBuilder argBuilder = new ArgumentBuilder().withName(name).withMinimum(1).withMaximum(1);

      if (defaultValue != null) {
        argBuilder = argBuilder.withDefault(defaultValue);
      }

      optBuilder.withArgument(argBuilder.create());
    }

    return optBuilder.create();
  }
  //convenience method

  /**
   *
   * @param name The name of the option
   * @return the {@link org.apache.commons.cli2.Option} with the name, else null
   */
  protected Option getCLIOption(String name){
    for (Option option : options) {
      if (option.getPreferredName().equals(name)){
        return option;
      }
    }
    return null;
  }

  /** Parse the arguments specified based on the options defined using the 
   *  various {@code addOption} methods. If -h is specified or an
   *  exception is encountered print help and return null. Has the 
   *  side effect of setting inputPath and outputPath 
   *  if {@code addInputOption} or {@code addOutputOption}
   *  or {@code mapred.input.dir} or {@code mapred.output.dir}
   *  are present in the Configuration.
   *
   * @return a {@code Map<String,String>} containing options and their argument values.
   *  The presence of a flag can be tested using {@code containsKey}, while
   *  argument values can be retrieved using {@code get(optionName)}. The
   *  names used for keys are the option name parameter prefixed by '--'.
   *
   *
   */
  public Map<String, String> parseArguments(String[] args) throws IOException {

    Option helpOpt = addOption(DefaultOptionCreator.helpOption());
    addOption("tempDir", null, "Intermediate output directory", "temp");
    addOption("startPhase", null, "First phase to run", "0");
    addOption("endPhase", null, "Last phase to run", String.valueOf(Integer.MAX_VALUE));

    GroupBuilder gBuilder = new GroupBuilder().withName("Job-Specific Options:");

    for (Option opt : options) {
      gBuilder = gBuilder.withOption(opt);
    }

    group = gBuilder.create();

    CommandLine cmdLine;
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      cmdLine = parser.parse(args);

    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelpWithGenericOptions(group, e);
      return null;
    }

    if (cmdLine.hasOption(helpOpt)) {
      CommandLineUtil.printHelpWithGenericOptions(group);
      return null;
    }

    try {
      parseDirectories(cmdLine);
    } catch (IllegalArgumentException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelpWithGenericOptions(group);
      return null;
    }

    argMap = new TreeMap<String, String>();
    maybePut(argMap, cmdLine, this.options.toArray(new Option[this.options.size()]));

    this.tempPath = new Path(argMap.get("--tempDir"));

    log.info("Command line arguments: {}", argMap);
    return argMap;
  }
  
  /**
   * Build the option key (--name) from the option name
   */
  public static String keyFor(String optionName) {
    return "--" + optionName;
  }

  /**
   * @return the requested option, or null if it has not been specified
   */
  public String getOption(String optionName) {
    return argMap.get(keyFor(optionName));
  }

  /**
   * @return if the requested option has been specified
   */
  public boolean hasOption(String optionName) {
    return argMap.containsKey(keyFor(optionName));
  }

  /** Obtain input and output directories from command-line options or hadoop
   *  properties. If {@code addInputOption} or {@code addOutputOption}
   *  has been called, this method will throw an {@code OptionException} if
   *  no source (command-line or property) for that value is present. 
   *  Otherwise, {@code inputPath} or {@code outputPath} will be
   *  non-null only if specified as a hadoop property. Command-line options
   *  take precedence over hadoop properties.
   *
   * @param cmdLine
   * @throws IllegalArgumentException if either inputOption is present,
   *   and neither {@code --input} nor {@code -Dmapred.input dir} are
   *   specified or outputOption is present and neither {@code --output}
   *   nor {@code -Dmapred.output.dir} are specified.
   */
  protected void parseDirectories(CommandLine cmdLine) {

    Configuration conf = getConf();

    if (inputOption != null && cmdLine.hasOption(inputOption)) {
      this.inputPath = new Path(cmdLine.getValue(inputOption).toString());
    }
    if (inputPath == null && conf.get("mapred.input.dir") != null) {
      this.inputPath = new Path(conf.get("mapred.input.dir"));
    }

    if (outputOption != null && cmdLine.hasOption(outputOption)) {
      this.outputPath = new Path(cmdLine.getValue(outputOption).toString());
    }
    if (outputPath == null && conf.get("mapred.output.dir") != null) {
      this.outputPath = new Path(conf.get("mapred.output.dir"));
    }

    Preconditions.checkArgument(inputOption == null || inputPath != null,
        "No input specified or -Dmapred.input.dir must be provided to specify input directory");
    Preconditions.checkArgument(outputOption == null || outputPath != null,
        "No output specified:  or -Dmapred.output.dir must be provided to specify output directory");
  }

  protected static void maybePut(Map<String, String> args, CommandLine cmdLine, Option... opt) {
    for (Option o : opt) {

      // the option appeared on the command-line, or it has a value
      // (which is likely a default value). 
      if (cmdLine.hasOption(o) || cmdLine.getValue(o) != null) {

        // nulls are ok, for cases where options are simple flags.
        Object vo = cmdLine.getValue(o);
        String value = vo == null ? null : vo.toString();
        args.put(o.getPreferredName(), value);
      }
    }
  }

  protected static boolean shouldRunNextPhase(Map<String, String> args, AtomicInteger currentPhase) {
    int phase = currentPhase.getAndIncrement();
    String startPhase = args.get("--startPhase");
    String endPhase = args.get("--endPhase");
    boolean phaseSkipped = (startPhase != null && phase < Integer.parseInt(startPhase))
        || (endPhase != null && phase > Integer.parseInt(endPhase));
    if (phaseSkipped) {
      log.info("Skipping phase {}", phase);
    }
    return !phaseSkipped;
  }

  protected Job prepareJob(Path inputPath,
                           Path outputPath,
                           Class<? extends InputFormat> inputFormat,
                           Class<? extends Mapper> mapper,
                           Class<? extends Writable> mapperKey,
                           Class<? extends Writable> mapperValue,
                           Class<? extends OutputFormat> outputFormat) throws IOException {

    Job job = HadoopUtil.prepareJob(inputPath, outputPath,
            inputFormat, mapper, mapperKey, mapperValue, outputFormat, getConf());
    job.setJobName(HadoopUtil.getCustomJobName(getClass().getSimpleName(), job, mapper, Reducer.class));
    return job;

  }

  protected Job prepareJob(Path inputPath, Path outputPath, Class<? extends Mapper> mapper,
      Class<? extends Writable> mapperKey, Class<? extends Writable> mapperValue, Class<? extends Reducer> reducer,
      Class<? extends Writable> reducerKey, Class<? extends Writable> reducerValue) throws IOException {
    return prepareJob(inputPath, outputPath, SequenceFileInputFormat.class, mapper, mapperKey, mapperValue, reducer,
        reducerKey, reducerValue, SequenceFileOutputFormat.class);
  }

  protected Job prepareJob(Path inputPath,
                           Path outputPath,
                           Class<? extends InputFormat> inputFormat,
                           Class<? extends Mapper> mapper,
                           Class<? extends Writable> mapperKey,
                           Class<? extends Writable> mapperValue,
                           Class<? extends Reducer> reducer,
                           Class<? extends Writable> reducerKey,
                           Class<? extends Writable> reducerValue,
                           Class<? extends OutputFormat> outputFormat) throws IOException {
    Job job = HadoopUtil.prepareJob(inputPath, outputPath,
            inputFormat, mapper, mapperKey, mapperValue, reducer, reducerKey, reducerValue, outputFormat, getConf());
    job.setJobName(HadoopUtil.getCustomJobName(getClass().getSimpleName(), job, mapper, Reducer.class));
    return job;
  }

  /**
   * necessary to make this job (having a combined input path) work on Amazon S3, hopefully this is obsolete when MultipleInputs is available
   * again
   */
  public static void setS3SafeCombinedInputPath(Job job, Path referencePath, Path inputPathOne, Path inputPathTwo)
      throws IOException {
    FileSystem fs = FileSystem.get(referencePath.toUri(), job.getConfiguration());
    FileInputFormat.setInputPaths(job, inputPathOne.makeQualified(fs), inputPathTwo.makeQualified(fs));
  }

  protected Class<? extends Analyzer> getAnalyzerClassFromOption() throws ClassNotFoundException {
    Class<? extends Analyzer> analyzerClass = DefaultAnalyzer.class;
    if (hasOption(DefaultOptionCreator.ANALYZER_NAME_OPTION)) {
      String className = getOption(DefaultOptionCreator.ANALYZER_NAME_OPTION);
      analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
      // try instantiating it, b/c there isn't any point in setting it if
      // you can't instantiate it
      ClassUtils.instantiateAs(analyzerClass, Analyzer.class);
    }
    return analyzerClass;
  }
}
