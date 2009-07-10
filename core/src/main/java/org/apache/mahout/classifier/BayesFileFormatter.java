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

package org.apache.mahout.classifier;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
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
 * Flatten a file into format that can be read by the Bayes M/R job. <p/> One document per line, first token is the
 * label followed by a tab, rest of the line are the terms.
 */
public class BayesFileFormatter {

  private static final Logger log = LoggerFactory.getLogger(BayesFileFormatter.class);

  private static final String LINE_SEP = System.getProperty("line.separator");

  private BayesFileFormatter() {
  }

  /**
   * Collapse all the files in the inputDir into a single file in the proper Bayes format, 1 document per line
   *
   * @param label      The label
   * @param analyzer   The analyzer to use
   * @param inputDir   The input Directory
   * @param charset    The charset of the input files
   * @param outputFile The file to collapse to
   */
  public static void collapse(String label, Analyzer analyzer, File inputDir,
                              Charset charset, File outputFile) throws IOException {
    Writer writer = new OutputStreamWriter(new FileOutputStream(outputFile),
        charset);
    try {
      inputDir.listFiles(new FileProcessor(label, analyzer, charset, writer));
      // listFiles() is called here as a way to recursively visit files, actually
    } finally {
      quietClose(writer);
    }
  }

  /**
   * Write the input files to the outdir, one output file per input file
   *
   * @param label    The label of the file
   * @param analyzer The analyzer to use
   * @param input    The input file or directory. May not be null
   * @param charset  The Character set of the input files
   * @param outDir   The output directory. Files will be written there with the same name as the input file
   */
  public static void format(String label, Analyzer analyzer, File input,
                            Charset charset, File outDir) throws IOException {
    if (input.isDirectory()) {
      input.listFiles(new FileProcessor(label, analyzer, charset, outDir));
    } else {
      Writer writer = new OutputStreamWriter(new FileOutputStream(new File(
          outDir, input.getName())), charset);
      try {
        writeFile(label, analyzer, input, charset, writer);
      } finally {
        quietClose(writer);
      }
    }
  }

  /**
   * Hack the FileFilter mechanism so that we don't get stuck on large directories and don't have to loop the list
   * twice
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
     * @param label  The label
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
     * @param outputDir must not be null.
     */
    private FileProcessor(String label, Analyzer analyzer, Charset charset,
                          File outputDir) {
      this.label = label;
      this.analyzer = analyzer;
      this.charset = charset;
      this.outputDir = outputDir;
    }

    @Override
    public boolean accept(File file) {
      if (file.isFile()) {
        Writer theWriter = null;
        try {
          if (writer == null) {
            theWriter = new OutputStreamWriter(new FileOutputStream(new File(
                outputDir, file.getName())), charset);
          } else {
            theWriter = writer;
          }
          writeFile(label, analyzer, file, charset, theWriter);
          if (writer != null) {
            // just write a new line
            theWriter.write(LINE_SEP);
          }
        } catch (IOException e) {
          // TODO: report failed files instead of throwing exception
          throw new RuntimeException(e);
        } finally {
          if (writer == null) {
            quietClose(theWriter);
          }
        }
      } else {
        file.listFiles(this);
      }
      return false;
    }
  }

  /**
   * Write the tokens and the label from the Reader to the writer
   *
   * @param label    The label
   * @param analyzer The analyzer to use
   * @param inFile   the file to read and whose contents are passed to the analyzer
   * @param charset  character encoding to assume when reading the input file
   * @param writer   The Writer, is not closed by this method
   * @throws java.io.IOException if there was a problem w/ the reader
   */
  private static void writeFile(String label, Analyzer analyzer, File inFile,
                                Charset charset, Writer writer) throws IOException {
    Reader reader = new InputStreamReader(new FileInputStream(inFile), charset);
    try {
      TokenStream ts = analyzer.tokenStream(label, reader);
      writer.write(label);
      writer.write('\t'); // edit: Inorder to match Hadoop standard
      // TextInputFormat
      Token token = new Token();
      while ((token = ts.next(token)) != null) {
        char[] termBuffer = token.termBuffer();
        int termLen = token.termLength();
        writer.write(termBuffer, 0, termLen);
        writer.write(' ');
      }
    } finally {
      quietClose(reader);
    }
  }

  private static void quietClose(Closeable closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (IOException ioe) {
        // continue
      }
    }
  }

  /**
   * Convert a Reader to a vector
   *
   * @param analyzer The Analyzer to use
   * @param reader   The reader to feed to the Analyzer
   * @return An array of unique tokens
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
   * @throws IOException            if the files can't be dealt with properly
   */
  public static void main(String[] args) throws ClassNotFoundException,
      IllegalAccessException, InstantiationException, IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
        withDescription("The Input file").withShortName("i").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
        withDescription("The output file").withShortName("o").create();

    Option labelOpt = obuilder.withLongName("label").withRequired(true).withArgument(
        abuilder.withName("label").withMinimum(1).withMaximum(1).create()).
        withDescription("The label of the file").withShortName("l").create();

    Option analyzerOpt = obuilder.withLongName("analyzer").withArgument(
        abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).
        withDescription("The fully qualified class name of the analyzer to use.  Must have a no-arg constructor.  Default is the StandardAnalyzer").withShortName("a").create();

    Option charsetOpt = obuilder.withLongName("charset").withArgument(
        abuilder.withName("charset").withMinimum(1).withMaximum(1).create()).
        withDescription("The character encoding of the input file").withShortName("c").create();

    Option collapseOpt = obuilder.withLongName("collapse").withRequired(true).withArgument(
        abuilder.withName("collapse").withMinimum(1).withMaximum(1).create()).
        withDescription("Collapse a whole directory to a single file, one doc per line").withShortName("p").create();

    Option helpOpt = obuilder.withLongName("help").withRequired(true).
        withDescription("Print out help").withShortName("h").create();
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(labelOpt).withOption(analyzerOpt).withOption(charsetOpt).withOption(collapseOpt).withOption(helpOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        return;
      }
      File input = new File((String) cmdLine.getValue(inputOpt));
      File output = new File((String) cmdLine.getValue(outputOpt));
      String label = (String) cmdLine.getValue(labelOpt);
      Analyzer analyzer;
      if (cmdLine.hasOption(analyzerOpt)) {
        analyzer = Class.forName(
            (String) cmdLine.getValue(analyzerOpt)).asSubclass(Analyzer.class).newInstance();
      } else {
        analyzer = new StandardAnalyzer();
      }
      Charset charset = Charset.forName("UTF-8");
      if (cmdLine.hasOption(charsetOpt)) {
        charset = Charset.forName((String) cmdLine.getValue(charsetOpt));
      }
      boolean collapse = cmdLine.hasOption(collapseOpt);

      if (collapse) {
        collapse(label, analyzer, input, charset, output);
      } else {
        format(label, analyzer, input, charset, output);
      }

    } catch (OptionException e) {
      log.error("Exception", e);
    }
  }
}
