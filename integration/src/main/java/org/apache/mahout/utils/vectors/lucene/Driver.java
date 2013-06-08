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

package org.apache.mahout.utils.vectors.lucene;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Iterator;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.io.DelimitedTermInfoWriter;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;
import org.apache.mahout.utils.vectors.io.VectorWriter;
import org.apache.mahout.vectorizer.TF;
import org.apache.mahout.vectorizer.TFIDF;
import org.apache.mahout.vectorizer.Weight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Driver {

  private static final Logger log = LoggerFactory.getLogger(Driver.class);

  private String luceneDir;
  private String outFile;
  private String field;
  private String idField;
  private String dictOut;
  private String seqDictOut = "";
  private String weightType = "tfidf";
  private String delimiter = "\t";
  private double norm = LuceneIterable.NO_NORMALIZING;
  private long maxDocs = Long.MAX_VALUE;
  private int minDf = 1;
  private int maxDFPercent = 99;
  private double maxPercentErrorDocs = 0.0;

  public void dumpVectors() throws IOException {

    File file = new File(luceneDir);
    Preconditions.checkArgument(file.isDirectory(),
                                "Lucene directory: " + file.getAbsolutePath()
                                    + " does not exist or is not a directory");
    Preconditions.checkArgument(maxDocs >= 0, "maxDocs must be >= 0");
    Preconditions.checkArgument(minDf >= 1, "minDf must be >= 1");
    Preconditions.checkArgument(maxDFPercent <= 99, "maxDFPercent must be <= 99");

    Directory dir = FSDirectory.open(file);
    IndexReader reader = DirectoryReader.open(dir);
    

    Weight weight;
    if ("tf".equalsIgnoreCase(weightType)) {
      weight = new TF();
    } else if ("tfidf".equalsIgnoreCase(weightType)) {
      weight = new TFIDF();
    } else {
      throw new IllegalArgumentException("Weight type " + weightType + " is not supported");
    }

    TermInfo termInfo = new CachedTermInfo(reader, field, minDf, maxDFPercent);
    
    LuceneIterable iterable;
    if (norm == LuceneIterable.NO_NORMALIZING) {
      iterable = new LuceneIterable(reader, idField, field, termInfo, weight, LuceneIterable.NO_NORMALIZING,
          maxPercentErrorDocs);
    } else {
      iterable = new LuceneIterable(reader, idField, field, termInfo, weight, norm, maxPercentErrorDocs);
    }

    log.info("Output File: {}", outFile);

    VectorWriter vectorWriter = getSeqFileWriter(outFile);
    try {
      long numDocs = vectorWriter.write(iterable, maxDocs);
      log.info("Wrote: {} vectors", numDocs);
    } finally {
      Closeables.close(vectorWriter, false);
    }

    File dictOutFile = new File(dictOut);
    log.info("Dictionary Output file: {}", dictOutFile);
    Writer writer = Files.newWriter(dictOutFile, Charsets.UTF_8);
    DelimitedTermInfoWriter tiWriter = new DelimitedTermInfoWriter(writer, delimiter, field);
    try {
      tiWriter.write(termInfo);
    } finally {
      Closeables.close(tiWriter, false);
    }

    if (!"".equals(seqDictOut)) {
      log.info("SequenceFile Dictionary Output file: {}", seqDictOut);

      Path path = new Path(seqDictOut);
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(conf);
      SequenceFile.Writer seqWriter = null;
      try {
        seqWriter = SequenceFile.createWriter(fs, conf, path, Text.class, IntWritable.class);
        Text term = new Text();
        IntWritable termIndex = new IntWritable();

        Iterator<TermEntry> termEntries = termInfo.getAllEntries();
        while (termEntries.hasNext()) {
          TermEntry termEntry = termEntries.next();
          term.set(termEntry.getTerm());
          termIndex.set(termEntry.getTermIdx());
          seqWriter.append(term, termIndex);
        }
      } finally {
        Closeables.close(seqWriter, false);
      }

    }
  }

  public static void main(String[] args) throws IOException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("dir").withRequired(true).withArgument(
        abuilder.withName("dir").withMinimum(1).withMaximum(1).create())
        .withDescription("The Lucene directory").withShortName("d").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription("The output file")
        .withShortName("o").create();

    Option fieldOpt = obuilder.withLongName("field").withRequired(true).withArgument(
        abuilder.withName("field").withMinimum(1).withMaximum(1).create()).withDescription(
        "The field in the index").withShortName("f").create();

    Option idFieldOpt = obuilder.withLongName("idField").withRequired(false).withArgument(
        abuilder.withName("idField").withMinimum(1).withMaximum(1).create()).withDescription(
        "The field in the index containing the index.  If null, then the Lucene internal doc "
            + "id is used which is prone to error if the underlying index changes").create();

    Option dictOutOpt = obuilder.withLongName("dictOut").withRequired(true).withArgument(
        abuilder.withName("dictOut").withMinimum(1).withMaximum(1).create()).withDescription(
        "The output of the dictionary").withShortName("t").create();

    Option seqDictOutOpt = obuilder.withLongName("seqDictOut").withRequired(false).withArgument(
        abuilder.withName("seqDictOut").withMinimum(1).withMaximum(1).create()).withDescription(
        "The output of the dictionary as sequence file").withShortName("st").create();

    Option weightOpt = obuilder.withLongName("weight").withRequired(false).withArgument(
        abuilder.withName("weight").withMinimum(1).withMaximum(1).create()).withDescription(
        "The kind of weight to use. Currently TF or TFIDF").withShortName("w").create();

    Option delimiterOpt = obuilder.withLongName("delimiter").withRequired(false).withArgument(
        abuilder.withName("delimiter").withMinimum(1).withMaximum(1).create()).withDescription(
        "The delimiter for outputting the dictionary").withShortName("l").create();

    Option powerOpt = obuilder.withLongName("norm").withRequired(false).withArgument(
        abuilder.withName("norm").withMinimum(1).withMaximum(1).create()).withDescription(
        "The norm to use, expressed as either a double or \"INF\" if you want to use the Infinite norm.  "
            + "Must be greater or equal to 0.  The default is not to normalize").withShortName("n").create();

    Option maxOpt = obuilder.withLongName("max").withRequired(false).withArgument(
        abuilder.withName("max").withMinimum(1).withMaximum(1).create()).withDescription(
        "The maximum number of vectors to output.  If not specified, then it will loop over all docs")
        .withShortName("m").create();

    Option minDFOpt = obuilder.withLongName("minDF").withRequired(false).withArgument(
        abuilder.withName("minDF").withMinimum(1).withMaximum(1).create()).withDescription(
        "The minimum document frequency.  Default is 1").withShortName("md").create();

    Option maxDFPercentOpt = obuilder.withLongName("maxDFPercent").withRequired(false).withArgument(
        abuilder.withName("maxDFPercent").withMinimum(1).withMaximum(1).create()).withDescription(
        "The max percentage of docs for the DF.  Can be used to remove really high frequency terms."
            + "  Expressed as an integer between 0 and 100. Default is 99.").withShortName("x").create();

    Option maxPercentErrorDocsOpt = obuilder.withLongName("maxPercentErrorDocs").withRequired(false).withArgument(
        abuilder.withName("maxPercentErrorDocs").withMinimum(1).withMaximum(1).create()).withDescription(
        "The max percentage of docs that can have a null term vector. These are noise document and can occur if the " 
            + "analyzer used strips out all terms in the target field. This percentage is expressed as a value "
            + "between 0 and 1. The default is 0.").withShortName("err").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(idFieldOpt).withOption(
        outputOpt).withOption(delimiterOpt).withOption(helpOpt).withOption(fieldOpt).withOption(maxOpt)
        .withOption(dictOutOpt).withOption(seqDictOutOpt).withOption(powerOpt).withOption(maxDFPercentOpt)
        .withOption(weightOpt).withOption(minDFOpt).withOption(maxPercentErrorDocsOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        CommandLineUtil.printHelp(group);
        return;
      }

      if (cmdLine.hasOption(inputOpt)) { // Lucene case
        Driver luceneDriver = new Driver();
        luceneDriver.setLuceneDir(cmdLine.getValue(inputOpt).toString());

        if (cmdLine.hasOption(maxOpt)) {
          luceneDriver.setMaxDocs(Long.parseLong(cmdLine.getValue(maxOpt).toString()));
        }

        if (cmdLine.hasOption(weightOpt)) {
          luceneDriver.setWeightType(cmdLine.getValue(weightOpt).toString());
        }

        luceneDriver.setField(cmdLine.getValue(fieldOpt).toString());

        if (cmdLine.hasOption(minDFOpt)) {
          luceneDriver.setMinDf(Integer.parseInt(cmdLine.getValue(minDFOpt).toString()));
        }

        if (cmdLine.hasOption(maxDFPercentOpt)) {
          luceneDriver.setMaxDFPercent(Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString()));
        }

        if (cmdLine.hasOption(powerOpt)) {
          String power = cmdLine.getValue(powerOpt).toString();
          if ("INF".equals(power)) {
            luceneDriver.setNorm(Double.POSITIVE_INFINITY);
          } else {
            luceneDriver.setNorm(Double.parseDouble(power));
          }
        }

        if (cmdLine.hasOption(idFieldOpt)) {
          luceneDriver.setIdField(cmdLine.getValue(idFieldOpt).toString());
        }

        if (cmdLine.hasOption(maxPercentErrorDocsOpt)) {
          luceneDriver.setMaxPercentErrorDocs(Double.parseDouble(cmdLine.getValue(maxPercentErrorDocsOpt).toString()));
        }

        luceneDriver.setOutFile(cmdLine.getValue(outputOpt).toString());

        luceneDriver.setDelimiter(cmdLine.hasOption(delimiterOpt) ? cmdLine.getValue(delimiterOpt).toString() : "\t");

        luceneDriver.setDictOut(cmdLine.getValue(dictOutOpt).toString());

        if (cmdLine.hasOption(seqDictOutOpt)) {
          luceneDriver.setSeqDictOut(cmdLine.getValue(seqDictOutOpt).toString());
        }

        luceneDriver.dumpVectors();
      }
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  private static VectorWriter getSeqFileWriter(String outFile) throws IOException {
    Path path = new Path(outFile);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    // TODO: Make this parameter driven

    SequenceFile.Writer seqWriter = SequenceFile.createWriter(fs, conf, path, LongWritable.class,
                                                              VectorWritable.class);

    return new SequenceFileVectorWriter(seqWriter);
  }

  public void setLuceneDir(String luceneDir) {
    this.luceneDir = luceneDir;
  }

  public void setMaxDocs(long maxDocs) {
    this.maxDocs = maxDocs;
  }

  public void setWeightType(String weightType) {
    this.weightType = weightType;
  }

  public void setField(String field) {
    this.field = field;
  }

  public void setMinDf(int minDf) {
    this.minDf = minDf;
  }

  public void setMaxDFPercent(int maxDFPercent) {
    this.maxDFPercent = maxDFPercent;
  }

  public void setNorm(double norm) {
    this.norm = norm;
  }

  public void setIdField(String idField) {
    this.idField = idField;
  }

  public void setOutFile(String outFile) {
    this.outFile = outFile;
  }

  public void setDelimiter(String delimiter) {
    this.delimiter = delimiter;
  }

  public void setDictOut(String dictOut) {
    this.dictOut = dictOut;
  }

  public void setSeqDictOut(String seqDictOut) {
    this.seqDictOut = seqDictOut;
  }

  public void setMaxPercentErrorDocs(double maxPercentErrorDocs) {
    this.maxPercentErrorDocs = maxPercentErrorDocs;
  }
}
