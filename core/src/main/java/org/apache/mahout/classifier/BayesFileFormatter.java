package org.apache.mahout.classifier;

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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.cli.Parser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * Flatten a file into format that can be read by the Bayes M/R job. <p/> One
 * document per line, first token is the label followed by a tab, rest of the
 * line are the terms.
 */
public class BayesFileFormatter {

  private static final Logger log = LoggerFactory.getLogger(BayesFileFormatter.class);

  private static final String LINE_SEP = System.getProperty("line.separator");

  /**
   * Collapse all the files in the inputDir into a single file in the proper
   * Bayes format, 1 document per line
   * 
   * @param label The label
   * @param analyzer The analyzer to use
   * @param inputDir The input Directory
   * @param charset The charset of the input files
   * @param outputFile The file to collapse to
   * @throws java.io.IOException
   */
  public static void collapse(String label, Analyzer analyzer, File inputDir,
      Charset charset, File outputFile) throws IOException {
    Writer writer = new OutputStreamWriter(new FileOutputStream(outputFile),
        charset);
    inputDir.listFiles(new FileProcessor(label, analyzer, charset, writer));
    // TODO srowen asks why call this when return value isn't used?
    writer.close();

  }

  /**
   * Write the input files to the outdir, one output file per input file
   * 
   * @param label The label of the file
   * @param analyzer The analyzer to use
   * @param input The input file or directory. May not be null
   * @param charset The Character set of the input files
   * @param outDir The output directory. Files will be written there with the
   *        same name as the input file
   * @throws IOException
   */
  public static void format(String label, Analyzer analyzer, File input,
      Charset charset, File outDir) throws IOException {
    if (input.isDirectory() == false) {
      Writer writer = new OutputStreamWriter(new FileOutputStream(new File(
          outDir, input.getName())), charset);
      writeFile(label, analyzer, new InputStreamReader(new FileInputStream(
          input), charset), writer);
      writer.close();
    } else {
      input.listFiles(new FileProcessor(label, analyzer, charset, outDir));
      // TODO srowen asks why call this when return value isn't used?
    }
  }

  /**
   * Hack the FileFilter mechanism so that we don't get stuck on large
   * directories and don't have to loop the list twice
   */
  private static class FileProcessor implements FileFilter {
    private final String label;

    private final Analyzer analyzer;

    private File outputDir;

    private final Charset charset;

    private Writer writer;

    /**
     * Use this when you want to collapse all files to a single file
     * 
     * @param label The label
     * @param analyzer
     * @param charset
     * @param writer must not be null and will not be closed
     */
    private FileProcessor(String label, Analyzer analyzer, Charset charset,
        Writer writer) {
      this.label = label;
      this.analyzer = analyzer;
      this.charset = charset;
      this.writer = writer;
    }

    /**
     * Use this when you want a writer per file
     * 
     * @param label
     * @param analyzer
     * @param charset
     * @param outputDir must not be null.
     */
    private FileProcessor(String label, Analyzer analyzer, Charset charset,
        File outputDir) {
      this.label = label;
      this.analyzer = analyzer;
      this.charset = charset;
      this.outputDir = outputDir;
    }

    public boolean accept(File file) {
      if (file.isFile()) {
        try {
          Writer theWriter;
          if (writer == null) {
            theWriter = new OutputStreamWriter(new FileOutputStream(new File(
                outputDir, file.getName())), charset);
          } else {
            theWriter = writer;
          }
          writeFile(label, analyzer, new InputStreamReader(new FileInputStream(
              file), charset), theWriter);
          if (writer == null) {
            theWriter.close();// we are writing a single file per input file
          } else {
            // just write a new line
            theWriter.write(LINE_SEP);

          }

        } catch (IOException e) {
          // TODO: report failed files instead of throwing exception
          throw new RuntimeException(e);
        }
      } else {
        file.listFiles(this);
        // TODO srowen asks why call this when return value isn't used?
      }
      return false;
    }
  }

  /**
   * Write the tokens and the label from the Reader to the writer
   * 
   * @param label The label
   * @param analyzer The analyzer to use
   * @param reader The reader to pass to the Analyzer
   * @param writer The Writer, is not closed by this method
   * @throws java.io.IOException if there was a problem w/ the reader
   */
  public static void writeFile(String label, Analyzer analyzer, Reader reader,
      Writer writer) throws IOException {
    TokenStream ts = analyzer.tokenStream(label, reader);
    writer.write(label);
    writer.write('\t'); // edit: Inorder to match Hadoop standard
    // TextInputFormat
    Token token = new Token();
    CharArraySet seen = new CharArraySet(256, false);
    // TODO srowen wonders that 'seen' is updated but not used?
    //long numTokens = 0;
    while ((token = ts.next(token)) != null) {
      char[] termBuffer = token.termBuffer();
      int termLen = token.termLength();   
       
      writer.write(termBuffer, 0, termLen);
      writer.write(' ');
      char[] tmp = new char[termLen];
      System.arraycopy(termBuffer, 0, tmp, 0, termLen);
      seen.add(tmp);// do this b/c CharArraySet doesn't allow offsets
    }
    ///numTokens++;

  }

  /**
   * Convert a Reader to a vector
   * 
   * @param analyzer The Analyzer to use
   * @param reader The reader to feed to the Analyzer
   * @return An array of unique tokens
   * @throws IOException
   */
  public static String[] readerToDocument(Analyzer analyzer, Reader reader)
      throws IOException {
    TokenStream ts = analyzer.tokenStream("", reader);

    Token token;
    List<String> coll = new ArrayList<String>();
    while ((token = ts.next()) != null) {
      char[] termBuffer = token.termBuffer();
      int termLen = token.termLength();
      String val = new String(termBuffer, 0, termLen);
      coll.add(val);
    }
    return coll.toArray(new String[coll.size()]);
  }

  /**
   * Run the FileFormatter
   * 
   * @param args The input args. Run with -h to see the help
   * @throws ClassNotFoundException if the Analyzer can't be found
   * @throws IllegalAccessException if the Analyzer can't be constructed
   * @throws InstantiationException if the Analyzer can't be constructed
   * @throws IOException if the files can't be dealt with properly
   */
  @SuppressWarnings("static-access")
  public static void main(String[] args) throws ClassNotFoundException,
      IllegalAccessException, InstantiationException, IOException {
    Options options = new Options();
    Option inputOpt = OptionBuilder.withLongOpt("input").isRequired().hasArg()
        .withDescription("The input file").create("i");
    options.addOption(inputOpt);
    Option outputOpt = OptionBuilder.withLongOpt("output").isRequired()
        .hasArg().withDescription("The output file").create("o");
    options.addOption(outputOpt);
    Option labelOpt = OptionBuilder.withLongOpt("label").isRequired().hasArg()
        .withDescription("The label of the file").create("l");
    options.addOption(labelOpt);
    Option analyzerOpt = OptionBuilder
        .withLongOpt("analyzer")
        .hasArg()
        .withDescription(
            "The fully qualified class name of the analyzer to use.  Must have a no-arg constructor.  Default is the StandardAnalyzer")
        .create("a");
    options.addOption(analyzerOpt);
    Option charsetOpt = OptionBuilder.withLongOpt("charset").hasArg()
        .withDescription("The character encoding of the input file")
        .create("c");
    options.addOption(charsetOpt);
    Option collapseOpt = OptionBuilder.withLongOpt("collapse").hasArg()
        .withDescription(
            "Collapse a whole directory to a single file, one doc per line")
        .create("p");
    options.addOption(collapseOpt);
    Option helpOpt = OptionBuilder.withLongOpt("help").withDescription(
        "Print out help info").create("h");
    options.addOption(helpOpt);
    CommandLine cmdLine;
    try {
      Parser parser = new PosixParser();
      cmdLine = parser.parse(options, args);
      if (cmdLine.hasOption(helpOpt.getOpt())) {
        log.info("Options: {}", options);
        return;
      }
      File input = new File(cmdLine.getOptionValue(inputOpt.getOpt()));
      File output = new File(cmdLine.getOptionValue(outputOpt.getOpt()));
      String label = cmdLine.getOptionValue(labelOpt.getOpt());
      Analyzer analyzer;
      if (cmdLine.hasOption(analyzerOpt.getOpt())) {
        analyzer = Class.forName(
            cmdLine.getOptionValue(analyzerOpt.getOpt())).asSubclass(Analyzer.class).newInstance();
      } else {
        analyzer = new StandardAnalyzer();
      }
      Charset charset = Charset.forName("UTF-8");
      if (cmdLine.hasOption(charsetOpt.getOpt())) {
        charset = Charset.forName(cmdLine.getOptionValue(charsetOpt.getOpt()));
      }
      boolean collapse = cmdLine.hasOption(collapseOpt.getOpt());

      if (collapse) {
        collapse(label, analyzer, input, charset, output);
      } else {
        format(label, analyzer, input, charset, output);
      }

    } catch (ParseException exp) {
      log.warn(exp.toString(), exp);
      log.info("Options: {}", options);
    }
  }
}
