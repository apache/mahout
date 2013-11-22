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

package org.apache.mahout.utils;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.commons.cli2.OptionException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.jet.random.sampling.RandomSampler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.BitSet;

/**
 * A utility for splitting files in the input format used by the Bayes
 * classifiers or anything else that has one item per line or SequenceFiles (key/value)
 * into training and test sets in order to perform cross-validation.
 * <p/>
 * <p/>
 * This class can be used to split directories of files or individual files into
 * training and test sets using a number of different methods.
 * <p/>
 * When executed via {@link #splitDirectory(Path)} or {@link #splitFile(Path)},
 * the lines read from one or more, input files are written to files of the same
 * name into the directories specified by the
 * {@link #setTestOutputDirectory(Path)} and
 * {@link #setTrainingOutputDirectory(Path)} methods.
 * <p/>
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
 * {@link RandomSampler} class from {@code mahout-math} is used to create a sample
 * of the appropriate size.</li>
 * </ul>
 * <p/>
 * Any one of the methods above can be used to control the size of the test set.
 * If multiple methods are called, a runtime exception will be thrown at
 * execution time.
 * <p/>
 * The {@link #setSplitLocation(int)} method is passed an integer from 0 to 100
 * (inclusive) which is translated into the position of the start of the test
 * data within the input file.
 * <p/>
 * Given:
 * <ul>
 * <li>an input file of 1500 lines</li>
 * <li>a desired test data size of 10 percent</li>
 * </ul>
 * <p/>
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
public class SplitInput extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(SplitInput.class);

  private int testSplitSize = -1;
  private int testSplitPct = -1;
  private int splitLocation = 100;
  private int testRandomSelectionSize = -1;
  private int testRandomSelectionPct = -1;
  private int keepPct = 100;
  private Charset charset = Charsets.UTF_8;
  private boolean useSequence;
  private boolean useMapRed;

  private Path inputDirectory;
  private Path trainingOutputDirectory;
  private Path testOutputDirectory;
  private Path mapRedOutputDirectory;

  private SplitCallback callback;

  @Override
  public int run(String[] args) throws Exception {

    if (parseArgs(args)) {
      splitDirectory();
    }
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new SplitInput(), args);
  }

  /**
   * Configure this instance based on the command-line arguments contained within provided array.
   * Calls {@link #validate()} to ensure consistency of configuration.
   *
   * @return true if the arguments were parsed successfully and execution should proceed.
   * @throws Exception if there is a problem parsing the command-line arguments or the particular
   *                   combination would violate class invariants.
   */
  private boolean parseArgs(String[] args) throws Exception {

    addInputOption();
    addOption("trainingOutput", "tr", "The training data output directory", false);
    addOption("testOutput", "te", "The test data output directory", false);
    addOption("testSplitSize", "ss", "The number of documents held back as test data for each category", false);
    addOption("testSplitPct", "sp", "The % of documents held back as test data for each category", false);
    addOption("splitLocation", "sl", "Location for start of test data expressed as a percentage of the input file "
        + "size (0=start, 50=middle, 100=end", false);
    addOption("randomSelectionSize", "rs", "The number of items to be randomly selected as test data ", false);
    addOption("randomSelectionPct", "rp", "Percentage of items to be randomly selected as test data when using "
        + "mapreduce mode", false);
    addOption("charset", "c", "The name of the character encoding of the input files (not needed if using "
        + "SequenceFiles)", false);
    addOption(buildOption("sequenceFiles", "seq", "Set if the input files are sequence files.  Default is false",
        false, false, "false"));
    addOption(DefaultOptionCreator.methodOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    //TODO: extend this to sequential mode
    addOption("keepPct", "k", "The percentage of total data to keep in map-reduce mode, the rest will be ignored.  "
        + "Default is 100%", false);
    addOption("mapRedOutputDir", "mro", "Output directory for map reduce jobs", false);

    if (parseArguments(args) == null) {
      return false;
    }

    try {
      inputDirectory = getInputPath();

      useMapRed = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(DefaultOptionCreator.MAPREDUCE_METHOD);

      if (useMapRed) {
        if (!hasOption("randomSelectionPct")) {
          throw new OptionException(getCLIOption("randomSelectionPct"),
                  "must set randomSelectionPct when mapRed option is used");
        }
        if (!hasOption("mapRedOutputDir")) {
          throw new OptionException(getCLIOption("mapRedOutputDir"),
                                    "mapRedOutputDir must be set when mapRed option is used");
        }
        mapRedOutputDirectory = new Path(getOption("mapRedOutputDir"));
        if (hasOption("keepPct")) {
          keepPct = Integer.parseInt(getOption("keepPct"));
        }
        if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
          HadoopUtil.delete(getConf(), mapRedOutputDirectory);
        }
      } else {
        if (!hasOption("trainingOutput")
                || !hasOption("testOutput")) {
          throw new OptionException(getCLIOption("trainingOutput"),
                  "trainingOutput and testOutput must be set if mapRed option is not used");
        }
        if (!hasOption("testSplitSize")
                && !hasOption("testSplitPct")
                && !hasOption("randomSelectionPct")
                && !hasOption("randomSelectionSize")) {
          throw new OptionException(getCLIOption("testSplitSize"),
                  "must set one of test split size/percentage or randomSelectionSize/percentage");
        }

        trainingOutputDirectory = new Path(getOption("trainingOutput"));
        testOutputDirectory = new Path(getOption("testOutput"));
        FileSystem fs = trainingOutputDirectory.getFileSystem(getConf());
        if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
          HadoopUtil.delete(fs.getConf(), trainingOutputDirectory);
          HadoopUtil.delete(fs.getConf(), testOutputDirectory);
        }
        fs.mkdirs(trainingOutputDirectory);
        fs.mkdirs(testOutputDirectory);
      }

      if (hasOption("charset")) {
        charset = Charset.forName(getOption("charset"));
      }

      if (hasOption("testSplitSize") && hasOption("testSplitPct")) {
        throw new OptionException(getCLIOption("testSplitPct"), "must have either split size or split percentage "
            + "option, not BOTH");
      }

      if (hasOption("testSplitSize")) {
        setTestSplitSize(Integer.parseInt(getOption("testSplitSize")));
      }

      if (hasOption("testSplitPct")) {
        setTestSplitPct(Integer.parseInt(getOption("testSplitPct")));
      }

      if (hasOption("splitLocation")) {
        setSplitLocation(Integer.parseInt(getOption("splitLocation")));
      }

      if (hasOption("randomSelectionSize")) {
        setTestRandomSelectionSize(Integer.parseInt(getOption("randomSelectionSize")));
      }

      if (hasOption("randomSelectionPct")) {
        setTestRandomSelectionPct(Integer.parseInt(getOption("randomSelectionPct")));
      }

      useSequence = hasOption("sequenceFiles");

    } catch (OptionException e) {
      log.error("Command-line option Exception", e);
      CommandLineUtil.printHelp(getGroup());
      return false;
    }

    validate();
    return true;
  }

  /**
   * Perform a split on directory specified by {@link #setInputDirectory(Path)} by calling {@link #splitFile(Path)}
   * on each file found within that directory.
   */
  public void splitDirectory() throws IOException, ClassNotFoundException, InterruptedException {
    this.splitDirectory(inputDirectory);
  }

  /**
   * Perform a split on the specified directory by calling {@link #splitFile(Path)} on each file found within that
   * directory.
   */
  public void splitDirectory(Path inputDir) throws IOException, ClassNotFoundException, InterruptedException {
    Configuration conf = getConf();
    splitDirectory(conf, inputDir);
  }

  /*
   * See also splitDirectory(Path inputDir)
   * */
  public void splitDirectory(Configuration conf, Path inputDir)
    throws IOException, ClassNotFoundException, InterruptedException {
    FileSystem fs = inputDir.getFileSystem(conf);
    if (fs.getFileStatus(inputDir) == null) {
      throw new IOException(inputDir + " does not exist");
    }
    if (!fs.getFileStatus(inputDir).isDir()) {
      throw new IOException(inputDir + " is not a directory");
    }

    if (useMapRed) {
      SplitInputJob.run(conf, inputDir, mapRedOutputDirectory,
            keepPct, testRandomSelectionPct);
    } else {
      // input dir contains one file per category.
      FileStatus[] fileStats = fs.listStatus(inputDir, PathFilters.logsCRCFilter());
      for (FileStatus inputFile : fileStats) {
        if (!inputFile.isDir()) {
          splitFile(inputFile.getPath());
        }
      }
    }
  }

  /**
   * Perform a split on the specified input file. Results will be written to files of the same name in the specified
   * training and test output directories. The {@link #validate()} method is called prior to executing the split.
   */
  public void splitFile(Path inputFile) throws IOException {
    Configuration conf = getConf();
    FileSystem fs = inputFile.getFileSystem(conf);
    if (fs.getFileStatus(inputFile) == null) {
      throw new IOException(inputFile + " does not exist");
    }
    if (fs.getFileStatus(inputFile).isDir()) {
      throw new IOException(inputFile + " is a directory");
    }

    validate();

    Path testOutputFile = new Path(testOutputDirectory, inputFile.getName());
    Path trainingOutputFile = new Path(trainingOutputDirectory, inputFile.getName());

    int lineCount = countLines(fs, inputFile, charset);

    log.info("{} has {} lines", inputFile.getName(), lineCount);

    int testSplitStart = 0;
    int testSplitSize = this.testSplitSize; // don't modify state
    BitSet randomSel = null;

    if (testRandomSelectionPct > 0 || testRandomSelectionSize > 0) {
      testSplitSize = this.testRandomSelectionSize;

      if (testRandomSelectionPct > 0) {
        testSplitSize = Math.round(lineCount * testRandomSelectionPct / 100.0f);
      }
      log.info("{} test split size is {} based on random selection percentage {}",
               inputFile.getName(), testSplitSize, testRandomSelectionPct);
      long[] ridx = new long[testSplitSize];
      RandomSampler.sample(testSplitSize, lineCount - 1, testSplitSize, 0, ridx, 0, RandomUtils.getRandom());
      randomSel = new BitSet(lineCount);
      for (long idx : ridx) {
        randomSel.set((int) idx + 1);
      }
    } else {
      if (testSplitPct > 0) { // calculate split size based on percentage
        testSplitSize = Math.round(lineCount * testSplitPct / 100.0f);
        log.info("{} test split size is {} based on percentage {}",
                 inputFile.getName(), testSplitSize, testSplitPct);
      } else {
        log.info("{} test split size is {}", inputFile.getName(), testSplitSize);
      }

      if (splitLocation > 0) { // calculate start of split based on percentage
        testSplitStart = Math.round(lineCount * splitLocation / 100.0f);
        if (lineCount - testSplitStart < testSplitSize) {
          // adjust split start downwards based on split size.
          testSplitStart = lineCount - testSplitSize;
        }
        log.info("{} test split start is {} based on split location {}",
                 inputFile.getName(), testSplitStart, splitLocation);
      }

      if (testSplitStart < 0) {
        throw new IllegalArgumentException("test split size for " + inputFile + " is too large, it would produce an "
                + "empty training set from the initial set of " + lineCount + " examples");
      } else if (lineCount - testSplitSize < testSplitSize) {
        log.warn("Test set size for {} may be too large, {} is larger than the number of "
                + "lines remaining in the training set: {}",
                 inputFile, testSplitSize, lineCount - testSplitSize);
      }
    }
    int trainCount = 0;
    int testCount = 0;
    if (!useSequence) {
      BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(inputFile), charset));
      Writer trainingWriter = new OutputStreamWriter(fs.create(trainingOutputFile), charset);
      Writer testWriter = new OutputStreamWriter(fs.create(testOutputFile), charset);


      try {

        String line;
        int pos = 0;
        while ((line = reader.readLine()) != null) {
          pos++;

          Writer writer;
          if (testRandomSelectionPct > 0) { // Randomly choose
            writer = randomSel.get(pos) ? testWriter : trainingWriter;
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

      } finally {
        Closeables.close(reader, true);
        Closeables.close(trainingWriter, false);
        Closeables.close(testWriter, false);
      }
    } else {
      SequenceFileIterator<Writable, Writable> iterator =
              new SequenceFileIterator<Writable, Writable>(inputFile, false, fs.getConf());
      SequenceFile.Writer trainingWriter = SequenceFile.createWriter(fs, fs.getConf(), trainingOutputFile,
          iterator.getKeyClass(), iterator.getValueClass());
      SequenceFile.Writer testWriter = SequenceFile.createWriter(fs, fs.getConf(), testOutputFile,
          iterator.getKeyClass(), iterator.getValueClass());
      try {

        int pos = 0;
        while (iterator.hasNext()) {
          pos++;
          SequenceFile.Writer writer;
          if (testRandomSelectionPct > 0) { // Randomly choose
            writer = randomSel.get(pos) ? testWriter : trainingWriter;
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
          Pair<Writable, Writable> pair = iterator.next();
          writer.append(pair.getFirst(), pair.getSecond());
        }

      } finally {
        Closeables.close(iterator, true);
        Closeables.close(trainingWriter, false);
        Closeables.close(testWriter, false);
      }
    }
    log.info("file: {}, input: {} train: {}, test: {} starting at {}",
             inputFile.getName(), lineCount, trainCount, testCount, testSplitStart);

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

  /**
   * Sets the percentage of the input data to allocate to the test split
   *
   * @param testSplitPct a value between 0 and 100 inclusive.
   */
  public void setTestSplitPct(int testSplitPct) {
    this.testSplitPct = testSplitPct;
  }

  /**
   * Sets the percentage of the input data to keep in a map reduce split input job
   *
   * @param keepPct a value between 0 and 100 inclusive.
   */
  public void setKeepPct(int keepPct) {
    this.keepPct = keepPct;
  }

  /**
   * Set to true to use map reduce to split the input
   *
   * @param useMapRed a boolean to indicate whether map reduce should be used
   */
  public void setUseMapRed(boolean useMapRed) {
    this.useMapRed = useMapRed;
  }

  public void setMapRedOutputDirectory(Path mapRedOutputDirectory) {
    this.mapRedOutputDirectory = mapRedOutputDirectory;
  }

  public int getSplitLocation() {
    return splitLocation;
  }

  /**
   * Set the location of the start of the test/training data split. Expressed as percentage of lines, for example
   * 0 indicates that the test data should be taken from the start of the file, 100 indicates that the test data
   * should be taken from the end of the input file, while 25 indicates that the test data should be taken from the
   * first quarter of the file.
   * <p/>
   * This option is only relevant in cases where random selection is not employed
   *
   * @param splitLocation a value between 0 and 100 inclusive.
   */
  public void setSplitLocation(int splitLocation) {
    this.splitLocation = splitLocation;
  }

  public Charset getCharset() {
    return charset;
  }

  /**
   * Set the charset used to read and write files
   */
  public void setCharset(Charset charset) {
    this.charset = charset;
  }

  public Path getInputDirectory() {
    return inputDirectory;
  }

  /**
   * Set the directory from which input data will be read when the the {@link #splitDirectory()} method is invoked
   */
  public void setInputDirectory(Path inputDir) {
    this.inputDirectory = inputDir;
  }

  public Path getTrainingOutputDirectory() {
    return trainingOutputDirectory;
  }

  /**
   * Set the directory to which training data will be written.
   */
  public void setTrainingOutputDirectory(Path trainingOutputDir) {
    this.trainingOutputDirectory = trainingOutputDir;
  }

  public Path getTestOutputDirectory() {
    return testOutputDirectory;
  }

  /**
   * Set the directory to which test data will be written.
   */
  public void setTestOutputDirectory(Path testOutputDir) {
    this.testOutputDirectory = testOutputDir;
  }

  public SplitCallback getCallback() {
    return callback;
  }

  /**
   * Sets the callback used to inform the caller that an input file has been successfully split
   */
  public void setCallback(SplitCallback callback) {
    this.callback = callback;
  }

  public int getTestRandomSelectionSize() {
    return testRandomSelectionSize;
  }

  /**
   * Sets number of random input samples that will be saved to the test set.
   */
  public void setTestRandomSelectionSize(int testRandomSelectionSize) {
    this.testRandomSelectionSize = testRandomSelectionSize;
  }

  public int getTestRandomSelectionPct() {

    return testRandomSelectionPct;
  }

  /**
   * Sets number of random input samples that will be saved to the test set as a percentage of the size of the
   * input set.
   *
   * @param randomSelectionPct a value between 0 and 100 inclusive.
   */
  public void setTestRandomSelectionPct(int randomSelectionPct) {
    this.testRandomSelectionPct = randomSelectionPct;
  }

  /**
   * Validates that the current instance is in a consistent state
   *
   * @throws IllegalArgumentException if settings violate class invariants.
   * @throws IOException              if output directories do not exist or are not directories.
   */
  public void validate() throws IOException {
    Preconditions.checkArgument(testSplitSize >= 1 || testSplitSize == -1,
        "Invalid testSplitSize: " + testSplitSize + ". Must be: testSplitSize >= 1 or testSplitSize = -1");
    Preconditions.checkArgument(splitLocation >= 0 && splitLocation <= 100 || splitLocation == -1,
        "Invalid splitLocation percentage: " + splitLocation + ". Must be: 0 <= splitLocation <= 100 or splitLocation = -1");
    Preconditions.checkArgument(testSplitPct >= 0 && testSplitPct <= 100 || testSplitPct == -1,
        "Invalid testSplitPct percentage: " + testSplitPct + ". Must be: 0 <= testSplitPct <= 100 or testSplitPct = -1");
    Preconditions.checkArgument(testRandomSelectionPct >= 0 && testRandomSelectionPct <= 100
            || testRandomSelectionPct == -1,"Invalid testRandomSelectionPct percentage: " + testRandomSelectionPct +
        ". Must be: 0 <= testRandomSelectionPct <= 100 or testRandomSelectionPct = -1");

    Preconditions.checkArgument(trainingOutputDirectory != null || useMapRed,
        "No training output directory was specified");
    Preconditions.checkArgument(testOutputDirectory != null || useMapRed, "No test output directory was specified");

    // only one of the following may be set, one must be set.
    int count = 0;
    if (testSplitSize > 0) {
      count++;
    }
    if (testSplitPct > 0) {
      count++;
    }
    if (testRandomSelectionSize > 0) {
      count++;
    }
    if (testRandomSelectionPct > 0) {
      count++;
    }

    Preconditions.checkArgument(count == 1, "Exactly one of testSplitSize, testSplitPct, testRandomSelectionSize, "
        + "testRandomSelectionPct should be set");

    if (!useMapRed) {
      Configuration conf = getConf();
      FileSystem fs = trainingOutputDirectory.getFileSystem(conf);
      FileStatus trainingOutputDirStatus = fs.getFileStatus(trainingOutputDirectory);
      Preconditions.checkArgument(trainingOutputDirStatus != null && trainingOutputDirStatus.isDir(),
          "%s is not a directory", trainingOutputDirectory);
      FileStatus testOutputDirStatus = fs.getFileStatus(testOutputDirectory);
      Preconditions.checkArgument(testOutputDirStatus != null && testOutputDirStatus.isDir(),
          "%s is not a directory", testOutputDirectory);
    }
  }

  /**
   * Count the lines in the file specified as returned by {@code BufferedReader.readLine()}
   *
   * @param inputFile the file whose lines will be counted
   * @param charset   the charset of the file to read
   * @return the number of lines in the input file.
   * @throws IOException if there is a problem opening or reading the file.
   */
  public static int countLines(FileSystem fs, Path inputFile, Charset charset) throws IOException {
    int lineCount = 0;
    BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(inputFile), charset));
    try {
      while (reader.readLine() != null) {
        lineCount++;
      }
    } finally {
      Closeables.close(reader, true);
    }

    return lineCount;
  }

  /**
   * Used to pass information back to a caller once a file has been split without the need for a data object
   */
  public interface SplitCallback {
    void splitComplete(Path inputFile, int lineCount, int trainCount, int testCount, int testSplitStart);
  }

}
