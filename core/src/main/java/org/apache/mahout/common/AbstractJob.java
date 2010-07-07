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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  private Path   inputPath; 
  
  /** output path, populated by {@link #parseArguments(String[]) */
  private Path   outputPath;
  
  /** internal list of options that have been added */
  private final List<Option> options;
  
  protected AbstractJob() {
    options = new LinkedList<Option>();
  }
  
  /** Returns the input path established by a call to {@link #parseArguments(String[])}.
   *  The source of the path may be an input option added using {@link #addInputOption()}
   *  or it may be the value of the <code>mapred.input.dir</code> configuration
   *  property. 
   * @return
   */
  protected Path getInputPath() {
    return inputPath;
  }
  
  /** Returns the output path established by a call to {@link #parseArguments(String[])}.
   *  The source of the path may be an output option added using {@link #addOutputOption()}
   *  or it may be the value of the <code>mapred.input.dir</code> configuration
   *  property. 
   * @return
   */
  protected Path getOutputPath() {
    return outputPath;
  }
  
  /** Add an option with no argument whose presence can be checked for using
   *  <code>containsKey<code> method on the map returned by 
   *  {@link #parseArguments(String[])};
   *  
   * @param name
   * @param shortName
   * @param description
   */
  protected void addFlag(String name, String shortName, String description) {
    options.add(buildOption(name, shortName, description, false, false, null));
  }
  
  /** Add an option to the the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called. This options has an argument
   *  with null as its default value.
   *  
   * @param name
   * @param shortName
   * @param description
   */
  protected void addOption(String name, String shortName, String description) {
    options.add(buildOption(name, shortName, description, true, false, null));
  }
  
  /** Add an option to the the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called.
   * 
   * @param name
   * @param shortName
   * @param description
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
   * @param name
   * @param shortName
   * @param description
   * @param defaultValue the default argument value if this argument is not
   *   found on the command-line. null is allowed.
   */
  protected void addOption(String name, String shortName, String description, String defaultValue) {
    options.add(buildOption(name, shortName, description, true, false, defaultValue));
  }

  /** Add an arbitrary option to the set of options this job will parse when
   *  {@link #parseArguments(String[])} is called. If this option has no
   *  argument, use <code>containsKey</code> on the map returned by 
   *  <code>parseArguments</code> to check for its presence. Otherwise, the
   *  string value of the option will be placed in the map using a key
   *  equal to this options long name preceded by '--'.
   * @param option
   * @return the option added.
   */
  protected Option addOption(Option option) {
    options.add(option);
    return option;
  }
  
  /** Add the default output directory option, '-o' which takes a directory
   *  name as an argument. When {@link #parseArguments(String[])} is 
   *  called, the outputPath will be set based upon the value for this option.
   *  This this method is called, the output is required. 
   */
  protected void addInputOption() {
    this.inputOption = addOption(DefaultOptionCreator.inputOption().create());
  }
  
  /** Add the default output directory option, '-o' which takes a directory
   *  name as an argument. When {@link #parseArguments(String[])} is 
   *  called, the outputPath will be set based upon the value for this option.
   *  This this method is called, the output is required. 
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
  private static Option buildOption(String name,
                                      String shortName,
                                      String description,
                                      boolean hasArg,
                                      boolean required,
                                      String defaultValue) {

    DefaultOptionBuilder optBuilder = new DefaultOptionBuilder()
      .withLongName(name)
      .withDescription(description)
      .withRequired(required);
      
    if (shortName != null) {
      optBuilder.withShortName(shortName);
    }
    
    if (hasArg) {
      ArgumentBuilder argBuilder = new ArgumentBuilder()
        .withName(name)
        .withMinimum(1)
        .withMaximum(1);
      
      if (defaultValue != null) {
        argBuilder = argBuilder.withDefault(defaultValue);
      }
      
      optBuilder.withArgument(argBuilder.create());
    }

    return optBuilder.create();
  }
  
  /** Parse the arguments specified based on the options defined using the 
   *  various <code>addOption</code> methods. If -h is specified or an 
   *  exception is encountered pring help and return null. Has the 
   *  side effect of setting inputPath and outputPath 
   *  if <code>addInputOption</code> or <code>addOutputOption</code> 
   *  or <code>mapred.input.dir</code> or <code>mapred.output.dir</code>
   *  are present in the Configuration.
   * 
   * @param args
   * @return a Map<String,Sting> containing options and their argument values.
   *  The presence of a flag can be tested using <code>containsKey</code>, while
   *  argument values can be retrieved using <code>get(optionName</code>. The
   *  names used for keys are the option name parameter prefixed by '--'.
   *  
   * 
   */
  public Map<String,String> parseArguments(String[] args) {
    
    Option helpOpt = addOption(DefaultOptionCreator.helpOption());
    addOption("tempDir", null, "Intermediate output directory", "temp");
    addOption("startPhase", null, "First phase to run", "0");
    addOption("endPhase", null, "Last phase to run", String.valueOf(Integer.MAX_VALUE));

    GroupBuilder gBuilder = new GroupBuilder().withName("Job-Specific Options:");
    
    for (Option opt : options) {
      gBuilder = gBuilder.withOption(opt);
    }
    
    Group group = gBuilder.create();
    
    CommandLine cmdLine;
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      cmdLine = parser.parse(args);
      
      
    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelpWithGenericOptions(group);
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
    
    
    Map<String,String> result = new TreeMap<String,String>();
    maybePut(result, cmdLine, this.options.toArray(new Option[this.options.size()]));

    log.info("Command line arguments: {}", result);
    return result;
  }
  
  /** Obtain input and output directories from command-line options or hadoop
   *  properties. If <code>addInputOption</code> or <code>addOutputOption</code>
   *  has been called, this method will throw an <code>OptionException</code> if
   *  no source (command-line or property) for that value is present. 
   *  Otherwise, <code>inputPath</code> or <code>outputPath<code> will be 
   *  non-null only if specified as a hadoop property. Command-line options
   *  take precedence over hadoop properties.
   * 
   * @param cmdLine
   * @throws IllegalArgumentException if either inputOption is present,
   *   and neither <code>--input</code> nor <code>-Dmapred.input dir</code> are 
   *   specified or outputOption is present and neither <code>--output</code> 
   *   nor <code>-Dmapred.output.dir</code> are specified.
   */
  protected void parseDirectories(CommandLine cmdLine) 
    throws IllegalArgumentException {
    
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
    
    if (inputOption != null && inputPath == null) {
      throw new IllegalArgumentException("No input specified: " +
          inputOption.getPreferredName() + " or -Dmapred.input.dir " +
          "must be provided to specify input directory");
    }
    
    if (outputOption != null && outputPath == null) {
      throw new IllegalArgumentException("No output specified: " +
          outputOption.getPreferredName() + " or -Dmapred.output.dir " +
          "must be provided to specify output directory");
    }
  }
  
  protected static void maybePut(Map<String,String> args, CommandLine cmdLine, Option... opt) {
    for (Option o : opt) {
      
      // the option appeared on the command-line, or it has a value
      // (which is likely a default value). 
      if (cmdLine.hasOption(o) || cmdLine.getValue(o) != null) {
        
        // nulls are ok, for cases where options are simple flags.
        Object vo = cmdLine.getValue(o);
        String value = (vo == null) ? null : vo.toString();
        args.put(o.getPreferredName(), value);
      }
    }
  }

  protected static boolean shouldRunNextPhase(Map<String,String> args, AtomicInteger currentPhase) {
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
                           Class<? extends Reducer> reducer,
                           Class<? extends Writable> reducerKey,
                           Class<? extends Writable> reducerValue,
                           Class<? extends OutputFormat> outputFormat) throws IOException {
    
    Job job = new Job(new Configuration(getConf()));
    Configuration jobConf = job.getConfiguration();

    if (reducer.equals(Reducer.class)) {
      if (mapper.equals(Mapper.class)) {
        throw new IllegalStateException("Can't figure out the user class jar file from mapper/reducer");
      }
      job.setJarByClass(mapper);
    } else {
      job.setJarByClass(reducer);
    }

    job.setInputFormatClass(inputFormat);
    jobConf.set("mapred.input.dir", inputPath.toString());

    job.setMapperClass(mapper);
    job.setMapOutputKeyClass(mapperKey);
    job.setMapOutputValueClass(mapperValue);

    jobConf.setBoolean("mapred.compress.map.output", true);

    job.setReducerClass(reducer);
    job.setOutputKeyClass(reducerKey);
    job.setOutputValueClass(reducerValue);

    String customJobName = job.getJobName();
    if (customJobName == null || customJobName.trim().length() == 0) {
      customJobName = getClass().getSimpleName();
    }
    customJobName += '-' + mapper.getSimpleName();
    customJobName += '-' + reducer.getSimpleName();
    job.setJobName(customJobName);

    job.setOutputFormatClass(outputFormat);
    jobConf.set("mapred.output.dir", outputPath.toString());
    
    return job;
  }
}
