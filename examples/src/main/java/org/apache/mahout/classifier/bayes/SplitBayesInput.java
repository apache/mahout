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

package org.apache.mahout.classifier.bayes;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.BitSet;

import com.google.common.base.Preconditions;
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
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.jet.random.sampling.RandomSampler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A utility for splitting files in the input format used by the Bayes
 * classifiers into training and test sets in order to perform cross-validation.
 * This class is not strictly confined to working with the Bayes classifier
 * input. It can be used for any input files where each line is a complete
 * sample.
 * <p>
 * This class can be used to split directories of files or individual files into
 * training and test sets using a number of different methods.
 * <p>
 * When executed via {@link #splitDirectory(Path)} or {@link #splitFile(Path)},
 * the lines read from one or more, input files are written to files of the same
 * name into the directories specified by the
 * {@link #setTestOutputDirectory(Path)} and
 * {@link #setTrainingOutputDirectory(Path)} methods.
 * <p>
 * The composition of the test set is determined using one of the following
 * approaches:
 * <ul>
 * <li>A contiguous set of items can be chosen from the input file(s) using the
 * {@link #setTestSplitSize(int)} or {@link #setTestSplitPct(int)} methods.
 * {@link #setTestSplitSize(int)} allocates a fixed number of items, while
 * {@link #setTestSplitPct(int)} allocates a percentage of the original input,
 * rounded up to the nearest integer. {@link #setSplitLocation(int)} is used to
 * control the position in the input from which the test data is extracted and
 * is described further below.</li>
 * <li>A random sampling of items can be chosen from the input files(s) using
 * the {@link #setTestRandomSelectionSize(int)} or
 * {@link #setTestRandomSelectionPct(int)} methods, each choosing a fixed test
 * set size or percentage of the input set size as described above. The
 * {@link org.apache.mahout.math.jet.random.sampling.RandomSampler
 * RandomSampler} class from <code>mahout-math</code> is used to create a sample
 * of the appropriate size.</li>
 * </ul>
 * <p>
 * Any one of the methods above can be used to control the size of the test set.
 * If multiple methods are called, a runtime exception will be thrown at
 * execution time.
 * <p>
 * The {@link #setSplitLocation(int)} method is passed an integer from 0 to 100
 * (inclusive) which is translated into the position of the start of the test
 * data within the input file.
 * <p>
 * Given:
 * <ul>
 * <li>an input file of 1500 lines</li>
 * <li>a desired test data size of 10 percent</li>
 * </ul>
 * <p>
 * <ul>
 * <li>A split location of 0 will cause the first 150 items appearing in the
 * input set to be written to the test set.</li>
 * <li>A split location of 25 will cause items 375-525 to be written to the test
 * set.</li>
 * <li>A split location of 100 will cause the last 150 items in the input to be
 * written to the test set</li>
 * </ul>
 * The start of the split will always be adjusted forwards in order to ensure
 * that the desired test set size is allocated. Split location has no effect is
 * random sampling is employed.
 */
public class SplitBayesInput {
  
  private static final Logger log = LoggerFactory.getLogger(SplitBayesInput.class);

  private int testSplitSize = -1;
  private int testSplitPct  = -1;
  private int splitLocation = 100;
  private int testRandomSelectionSize = -1;
  private int testRandomSelectionPct = -1;
  private Charset charset = Charset.forName("UTF-8");

  private final FileSystem fs;
  private Path inputDirectory;
  private Path trainingOutputDirectory;
  private Path testOutputDirectory;
  
  private SplitCallback callback;
  
  public static void main(String[] args) throws Exception {
    SplitBayesInput si = new SplitBayesInput();
    if (si.parseArgs(args)) {
      si.splitDirectory();
    }
  }
  
  public SplitBayesInput() throws IOException {
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }
  
  /** Configure this instance based on the command-line arguments contained within provided array. 
   * Calls {@link #validate()} to ensure consistency of configuration.
   * 
   * @return true if the arguments were parsed successfully and execution should proceed.
   * @throws Exception if there is a problem parsing the command-line arguments or the particular
   *   combination would violate class invariants.
   */
  public boolean parseArgs(String[] args) throws Exception {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Option inputDirOpt = obuilder.withLongName("inputDir").withRequired(true).withArgument(
        abuilder.withName("inputDir").withMinimum(1).withMaximum(1).create()).withDescription(
        "The input directory").withShortName("i").create();
    
    Option trainingOutputDirOpt = obuilder.withLongName("trainingOutputDir").withRequired(true).withArgument(
        abuilder.withName("outputDir").withMinimum(1).withMaximum(1).create()).withDescription(
        "The training data output directory").withShortName("tr").create();
    
    Option testOutputDirOpt = obuilder.withLongName("testOutputDir").withRequired(true).withArgument(
        abuilder.withName("outputDir").withMinimum(1).withMaximum(1).create()).withDescription(
        "The test data output directory").withShortName("te").create();
    
    Option testSplitSizeOpt = obuilder.withLongName("testSplitSize").withRequired(false).withArgument(
        abuilder.withName("splitSize").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of documents held back as test data for each category").withShortName("ss").create();
    
    Option testSplitPctOpt = obuilder.withLongName("testSplitPct").withRequired(false).withArgument(
        abuilder.withName("splitPct").withMinimum(1).withMaximum(1).create()).withDescription(
        "The percentage of documents held back as test data for each category").withShortName("sp").create();
    
    Option splitLocationOpt = obuilder.withLongName("splitLocation").withRequired(false).withArgument(
        abuilder.withName("splitLoc").withMinimum(1).withMaximum(1).create()).withDescription(
        "Location for start of test data expressed as a percentage of the input file size (0=start, 50=middle, 100=end")
        .withShortName("sl").create();
    
    Option randomSelectionSizeOpt = obuilder.withLongName("randomSelectionSize").withRequired(false).withArgument(
        abuilder.withName("randomSize").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of itemr to be randomly selected as test data ").withShortName("rs").create();
    
    Option randomSelectionPctOpt = obuilder.withLongName("randomSelectionPct").withRequired(false).withArgument(
        abuilder.withName("randomPct").withMinimum(1).withMaximum(1).create()).withDescription(
        "Percentage of items to be randomly selected as test data ").withShortName("rp").create();
    
    Option charsetOpt = obuilder.withLongName("charset").withRequired(true).withArgument(
        abuilder.withName("charset").withMinimum(1).withMaximum(1).create()).withDescription(
        "The name of the character encoding of the input files").withShortName("c").create();
    
    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(trainingOutputDirOpt)
         .withOption(testOutputDirOpt).withOption(testSplitSizeOpt).withOption(testSplitPctOpt)
         .withOption(splitLocationOpt).withOption(randomSelectionSizeOpt).withOption(randomSelectionPctOpt)
         .withOption(charsetOpt).create();
    
    try {
      
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return false;
      }
      
      inputDirectory = new Path((String) cmdLine.getValue(inputDirOpt));
      trainingOutputDirectory = new Path((String) cmdLine.getValue(trainingOutputDirOpt));
      testOutputDirectory = new Path((String) cmdLine.getValue(testOutputDirOpt));
     
      charset = Charset.forName((String) cmdLine.getValue(charsetOpt));

      if (cmdLine.hasOption(testSplitSizeOpt) && cmdLine.hasOption(testSplitPctOpt)) {
        throw new OptionException(testSplitSizeOpt, "must have either split size or split percentage option, not BOTH");
      } else if (!cmdLine.hasOption(testSplitSizeOpt) && !cmdLine.hasOption(testSplitPctOpt)) {
        throw new OptionException(testSplitSizeOpt, "must have either split size or split percentage option");
      }

      if (cmdLine.hasOption(testSplitSizeOpt)) {
        setTestSplitSize(Integer.parseInt((String) cmdLine.getValue(testSplitSizeOpt)));
      }
      
      if (cmdLine.hasOption(testSplitPctOpt)) {
        setTestSplitPct(Integer.parseInt((String) cmdLine.getValue(testSplitPctOpt)));
      }
      
      if (cmdLine.hasOption(splitLocationOpt)) {
        setSplitLocation(Integer.parseInt((String) cmdLine.getValue(splitLocationOpt)));
      }
      
      if (cmdLine.hasOption(randomSelectionSizeOpt)) {
        setTestRandomSelectionSize(Integer.parseInt((String) cmdLine.getValue(randomSelectionSizeOpt)));
      }
      
      if (cmdLine.hasOption(randomSelectionPctOpt)) {
        setTestRandomSelectionPct(Integer.parseInt((String) cmdLine.getValue(randomSelectionPctOpt)));
      }

      fs.mkdirs(trainingOutputDirectory);
      fs.mkdirs(testOutputDirectory);
     
    } catch (OptionException e) {
      log.error("Command-line option Exception", e);
      CommandLineUtil.printHelp(group);
      return false;
    }
    
    validate();
    return true;
  }
  
  /** Perform a split on directory specified by {@link #setInputDirectory(Path)} by calling {@link #splitFile(Path)}
   *  on each file found within that directory.
   */
  public void splitDirectory() throws IOException {
    this.splitDirectory(inputDirectory);
  }
  
  /** Perform a split on the specified directory by calling {@link #splitFile(Path)} on each file found within that
   *  directory.
   */
  public void splitDirectory(Path inputDir) throws IOException {
    if (fs.getFileStatus(inputDir) == null) {
      throw new IOException(inputDir + " does not exist");
    }
    else if (!fs.getFileStatus(inputDir).isDir()) {
      throw new IOException(inputDir + " is not a directory");
    }

    // input dir contains one file per category.
    FileStatus[] fileStats = fs.listStatus(inputDir);
    for (FileStatus inputFile : fileStats) {
      if (!inputFile.isDir()) {
        splitFile(inputFile.getPath());
      }
    }
  }
  

  /** Perform a split on the specified input file. Results will be written to files of the same name in the specified 
   *  training and test output directories. The {@link #validate()} method is called prior to executing the split.
   */
  public void splitFile(Path inputFile) throws IOException {
    if (fs.getFileStatus(inputFile) == null) {
      throw new IOException(inputFile + " does not exist");
    }
    else if (fs.getFileStatus(inputFile).isDir()) {
      throw new IOException(inputFile + " is a directory");
    }
    
    validate();
    
    Path testOutputFile = new Path(testOutputDirectory, inputFile.getName());
    Path trainingOutputFile = new Path(trainingOutputDirectory, inputFile.getName());
    
    int lineCount = countLines(fs, inputFile, charset);
    
    log.info("{} has {} lines", inputFile.getName(), lineCount);
    
    int testSplitStart = 0;
    int testSplitSize  = this.testSplitSize; // don't modify state
    BitSet randomSel = null;
    
    if (testRandomSelectionPct > 0 || testRandomSelectionSize > 0) {
      testSplitSize = this.testRandomSelectionSize;
      
      if (testRandomSelectionPct > 0) {
        testSplitSize = Math.round(lineCount * (testRandomSelectionPct / 100.0f));
      }
      log.info("{} test split size is {} based on random selection percentage {}",
               new Object[] {inputFile.getName(), testSplitSize, testRandomSelectionPct});
      long[] ridx = new long[testSplitSize];
      RandomSampler.sample(testSplitSize, lineCount - 1, testSplitSize, 0, ridx, 0, RandomUtils.getRandom());
      randomSel = new BitSet(lineCount);
      for (long idx : ridx) {
        randomSel.set((int) idx + 1);
      }
    } else {
      if (testSplitPct > 0) { // calculate split size based on percentage
        testSplitSize = Math.round(lineCount * (testSplitPct / 100.0f));
        log.info("{} test split size is {} based on percentage {}",
                 new Object[] {inputFile.getName(), testSplitSize, testSplitPct});
      } else {
        log.info("{} test split size is {}", inputFile.getName(), testSplitSize);
      }
      
      if (splitLocation > 0) { // calculate start of split based on percentage
        testSplitStart =  Math.round(lineCount * (splitLocation / 100.0f));
        if (lineCount - testSplitStart < testSplitSize) {
          // adjust split start downwards based on split size.
          testSplitStart = lineCount - testSplitSize;
        }
        log.info("{} test split start is {} based on split location {}",
                 new Object[] {inputFile.getName(), testSplitStart, splitLocation});
      }
      
      if (testSplitStart < 0) {
        throw new IllegalArgumentException("test split size for " + inputFile + " is too large, it would produce an "
            + "empty training set from the initial set of " + lineCount + " examples");
      } else if ((lineCount - testSplitSize) < testSplitSize) {
        log.warn("Test set size for {} may be too large, {} is larger than the number of "
                 + "lines remaining in the training set: {}",
                 new Object[] {inputFile, testSplitSize, lineCount - testSplitSize});
      }
    }
    
    BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(inputFile), charset));
    Writer trainingWriter = new OutputStreamWriter(fs.create(trainingOutputFile), charset);
    Writer testWriter     = new OutputStreamWriter(fs.create(testOutputFile), charset);

    int pos = 0;
    int trainCount = 0;
    int testCount = 0;

    String line;
    while ((line = reader.readLine()) != null) {
      pos++;

      Writer writer;
      if (testRandomSelectionPct > 0) { // Randomly choose
        writer =  randomSel.get(pos) ? testWriter : trainingWriter;
      } else { // Choose based on location
        writer = pos > testSplitStart ? testWriter : trainingWriter;
      }

      if (writer == testWriter) {
        if (testCount >= testSplitSize) {
          writer = trainingWriter;
        } else {
          testCount++;
        }
      }
      
      if (writer == trainingWriter) {
        trainCount++;
      }
      
      writer.write(line);
      writer.write('\n');
    }
    
    IOUtils.quietClose(trainingWriter);
    IOUtils.quietClose(testWriter);
    
    log.info("file: {}, input: {} train: {}, test: {} starting at {}",
             new Object[] {inputFile.getName(), lineCount, trainCount, testCount, testSplitStart});
    
    // testing;
    if (callback != null) {
      callback.splitComplete(inputFile, lineCount, trainCount, testCount, testSplitStart);
    }
  }
  
  public int getTestSplitSize() {
    return testSplitSize;
  }

  public void setTestSplitSize(int testSplitSize) {
    this.testSplitSize = testSplitSize;
  }

  public int getTestSplitPct() {
    return testSplitPct;
  }

  /** Sets the percentage of the input data to allocate to the test split 
   * 
   * @param testSplitPct 
   *   a value between 0 and 100 inclusive.
   */
  public void setTestSplitPct(int testSplitPct) {
    this.testSplitPct = testSplitPct;
  }

  public int getSplitLocation() {
    return splitLocation;
  }

  /** Set the location of the start of the test/training data split. Expressed as percentage of lines, for example
   *  0 indicates that the test data should be taken from the start of the file, 100 indicates that the test data 
   *  should be taken from the end of the input file, while 25 indicates that the test data should be taken from the 
   *  first quarter of the file. 
   *  <p>
   *  This option is only relevant in cases where random selection is not employed
   * 
   * @param splitLocation 
   *   a value between 0 and 100 inclusive.
   */
  public void setSplitLocation(int splitLocation) {
    this.splitLocation = splitLocation;
  }

  public Charset getCharset() {
    return charset;
  }

  /** Set the charset used to read and write files
   */
  public void setCharset(Charset charset) {
    this.charset = charset;
  }

  public Path getInputDirectory() {
    return inputDirectory;
  }

  /** Set the directory from which input data will be read when the the {@link #splitDirectory()} method is invoked
   */
  public void setInputDirectory(Path inputDir) {
    this.inputDirectory = inputDir;
  }

  public Path getTrainingOutputDirectory() {
    return trainingOutputDirectory;
  }

  /** Set the directory to which training data will be written.
   */
  public void setTrainingOutputDirectory(Path trainingOutputDir) {
    this.trainingOutputDirectory = trainingOutputDir;
  }

  public Path getTestOutputDirectory() {
    return testOutputDirectory;
  }

  /** Set the directory to which test data will be written.
   */
  public void setTestOutputDirectory(Path testOutputDir) {
    this.testOutputDirectory = testOutputDir;
  }

  public SplitCallback getCallback() {
    return callback;
  }

  /** Sets the callback used to inform the caller that an input file has been successfully split
   */
  public void setCallback(SplitCallback callback) {
    this.callback = callback;
  }

  public int getTestRandomSelectionSize() {
    return testRandomSelectionSize;
  }

  /** Sets number of random input samples that will be saved to the test set.
   */
  public void setTestRandomSelectionSize(int testRandomSelectionSize) {
    this.testRandomSelectionSize = testRandomSelectionSize;
  }

  public int getTestRandomSelectionPct() {

    return testRandomSelectionPct;
  }

  /** Sets number of random input samples that will be saved to the test set as a percentage of the size of the 
   *  input set.
   * 
   * @param randomSelectionPct a value between 0 and 100 inclusive.
   */
  public void setTestRandomSelectionPct(int randomSelectionPct) {
    this.testRandomSelectionPct = randomSelectionPct;
  }

  /** Validates that the current instance is in a consistent state
   * 
   * @throws IllegalArgumentException
   *   if settings violate class invariants.
   * @throws IOException 
   *   if output directories do not exist or are not directories.
   */
  public void validate() throws IOException {
    Preconditions.checkArgument(testSplitSize >= 1 || testSplitSize == -1,
                                "Invalid testSplitSize", testSplitSize);
    Preconditions.checkArgument((splitLocation >= 0 && splitLocation <= 100) || splitLocation == -1,
                                "Invalid splitLocation percentage", splitLocation);
    Preconditions.checkArgument((testSplitPct >= 0 && testSplitPct <= 100) || testSplitPct == -1,
                                "Invalid testSplitPct percentage", testSplitPct);
    Preconditions.checkArgument((splitLocation >= 0 && splitLocation <= 100) || splitLocation == -1,
                                "Invalid splitLocation percentage", splitLocation);
    Preconditions.checkArgument((testRandomSelectionPct >= 0 && testRandomSelectionPct <= 100)
                                || testRandomSelectionPct == -1,
                                "Invalid testRandomSelectionPct percentage", testRandomSelectionPct);

    Preconditions.checkArgument(trainingOutputDirectory != null, "No training output directory was specified");
    Preconditions.checkArgument(testOutputDirectory != null, "No test output directory was specified");

    // only one of the following may be set, one must be set.
    int count = 0;
    if (testSplitSize > 0) {
      count++;
    }
    if (testSplitPct  > 0) {
      count++;
    }
    if (testRandomSelectionSize > 0) {
      count++;
    }
    if (testRandomSelectionPct > 0) {
      count++;
    }

    Preconditions.checkArgument(count == 1,
        "Exactly one of testSplitSize, testSplitPct, testRandomSelectionSize, testRandomSelectionPct should be set");

    FileStatus trainingOutputDirStatus = fs.getFileStatus(trainingOutputDirectory);
    Preconditions.checkArgument(trainingOutputDirStatus != null && trainingOutputDirStatus.isDir(),
                                "%s is not a directory", trainingOutputDirectory);
    FileStatus testOutputDirStatus = fs.getFileStatus(testOutputDirectory);
    Preconditions.checkArgument(testOutputDirStatus != null && testOutputDirStatus.isDir(),
                                "%s is not a directory", testOutputDirectory);
  }
  
  /** Count the lines in the file specified as returned by <code>BufferedReader.readLine()</code>
   * 
   * @param inputFile 
   *   the file whose lines will be counted
   *   
   * @param charset
   *   the charset of the file to read
   *   
   * @return the number of lines in the input file.
   * 
   * @throws IOException 
   *   if there is a problem opening or reading the file.
   */
  public static int countLines(FileSystem fs, Path inputFile, Charset charset) throws IOException {
    int lineCount = 0;
    BufferedReader countReader = new BufferedReader(new InputStreamReader(fs.open(inputFile), charset));
    try {
      while (countReader.readLine() != null) {
        lineCount++;
      }
    } finally {
      IOUtils.quietClose(countReader);
    }
    
    return lineCount;
  }
  
  /** Used to pass information back to a caller once a file has been split without the need for a data object */
  public interface SplitCallback {
    void splitComplete(Path inputFile, int lineCount, int trainCount, int testCount, int testSplitStart);
  }

}
