package org.apache.mahout.utils.vectors.lucene;
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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.utils.CommandLineUtil;
import org.apache.mahout.utils.vectors.TF;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.io.JWriterTermInfoWriter;
import org.apache.mahout.utils.vectors.io.JWriterVectorWriter;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;
import org.apache.mahout.utils.vectors.io.VectorWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;


/**
 *
 *
 **/
public class Driver {
  private transient static Logger log = LoggerFactory.getLogger(Driver.class);

  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("dir").withRequired(true).withArgument(
            abuilder.withName("dir").withMinimum(1).withMaximum(1).create()).
            withDescription("The Lucene directory").withShortName("d").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file").withShortName("o").create();

    Option fieldOpt = obuilder.withLongName("field").withRequired(true).withArgument(
            abuilder.withName("field").withMinimum(1).withMaximum(1).create()).
            withDescription("The field in the index").withShortName("f").create();

    Option idFieldOpt = obuilder.withLongName("idField").withRequired(false).withArgument(
            abuilder.withName("idField").withMinimum(1).withMaximum(1).create()).
            withDescription("The field in the index containing the index.  If null, then the Lucene internal doc " +
                    "id is used which is prone to error if the underlying index changes").withShortName("i").create();

    Option dictOutOpt = obuilder.withLongName("dictOut").withRequired(true).withArgument(
            abuilder.withName("dictOut").withMinimum(1).withMaximum(1).create()).
            withDescription("The output of the dictionary").withShortName("t").create();

    Option weightOpt = obuilder.withLongName("weight").withRequired(false).withArgument(
            abuilder.withName("weight").withMinimum(1).withMaximum(1).create()).
            withDescription("The kind of weight to use. Currently TF or TFIDF").withShortName("w").create();

    Option delimiterOpt = obuilder.withLongName("delimiter").withRequired(false).withArgument(
            abuilder.withName("delimiter").withMinimum(1).withMaximum(1).create()).
            withDescription("The delimiter for outputing the dictionary").withShortName("l").create();
    Option powerOpt = obuilder.withLongName("norm").withRequired(false).withArgument(
            abuilder.withName("norm").withMinimum(1).withMaximum(1).create()).
            withDescription("The norm to use, expressed as either a double or \"INF\" if you want to use the Infinite norm.  " +
                    "Must be greater or equal to 0.  The default is not to normalize").withShortName("n").create();
    Option maxOpt = obuilder.withLongName("max").withRequired(false).withArgument(
            abuilder.withName("max").withMinimum(1).withMaximum(1).create()).
            withDescription("The maximum number of vectors to output.  If not specified, then it will loop over all docs").withShortName("m").create();

    Option outWriterOpt = obuilder.withLongName("outputWriter").withRequired(false).withArgument(
            abuilder.withName("outputWriter").withMinimum(1).withMaximum(1).create()).
            withDescription("The VectorWriter to use, either seq (SequenceFileVectorWriter - default) or file (Writes to a File using JSON format)").withShortName("e").create();
    Option minDFOpt = obuilder.withLongName("minDF").withRequired(false).withArgument(
            abuilder.withName("minDF").withMinimum(1).withMaximum(1).create()).
            withDescription("The minimum document frequency.  Default is 1").withShortName("md").create();
    Option maxDFPercentOpt = obuilder.withLongName("maxDFPercent").withRequired(false).withArgument(
            abuilder.withName("maxDFPercent").withMinimum(1).withMaximum(1).create()).
            withDescription("The max percentage of docs for the DF.  Can be used to remove really high frequency terms.  Expressed as an integer between 0 and 100. Default is 99.").withShortName("x").create();
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(idFieldOpt).withOption(outputOpt).withOption(delimiterOpt)
            .withOption(helpOpt).withOption(fieldOpt).withOption(maxOpt).withOption(dictOutOpt).withOption(powerOpt).withOption(outWriterOpt).withOption(maxDFPercentOpt)
            .withOption(weightOpt).withOption(minDFOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        CommandLineUtil.printHelp(group);
        return;
      }
      //Springify all this
      if (cmdLine.hasOption(inputOpt)) {//Lucene case
        File file = new File(cmdLine.getValue(inputOpt).toString());
        if (file.exists() && file.isDirectory()) {
          long maxDocs = Long.MAX_VALUE;
          if (cmdLine.hasOption(maxOpt)) {
            maxDocs = Long.parseLong(cmdLine.getValue(maxOpt).toString());
          }
          if (maxDocs < 0) {
            throw new IllegalArgumentException("maxDocs must be >= 0");
          }
          Directory dir = FSDirectory.open(file);
          IndexReader reader = IndexReader.open(dir, true);
          Weight weight = null;
          if (cmdLine.hasOption(weightOpt)) {
            String wString = cmdLine.getValue(weightOpt).toString();
            if (wString.equalsIgnoreCase("tf")) {
              weight = new TF();
            } else if (wString.equalsIgnoreCase("tfidf")) {
              weight = new TFIDF();
            } else {
              throw new OptionException(weightOpt);
            }
          } else {
            weight = new TFIDF();
          }
          String field = cmdLine.getValue(fieldOpt).toString();
          int minDf = 1;
          if (cmdLine.hasOption(minDFOpt)) {
            minDf = Integer.parseInt(cmdLine.getValue(minDFOpt).toString());
          }
          int maxDFPercent = 99;
          if (cmdLine.hasOption(maxDFPercentOpt)) {
            maxDFPercent = Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString());
          }
          TermInfo termInfo = new CachedTermInfo(reader, field, minDf, maxDFPercent);
          VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
          LuceneIterable iterable = null;
          String power = null;
          double norm = -1;
          if (cmdLine.hasOption(powerOpt)) {
            power = cmdLine.getValue(powerOpt).toString();
            if (power.equals("INF")) {
              norm = Double.POSITIVE_INFINITY;
            } else {
              norm = Double.parseDouble(power);
            }
          }
          String idField = null;
          if (cmdLine.hasOption(idFieldOpt)) {
            idField = cmdLine.getValue(idFieldOpt).toString();
          }
          if (norm == LuceneIterable.NO_NORMALIZING) {
            iterable = new LuceneIterable(reader, idField, field, mapper, LuceneIterable.NO_NORMALIZING);
          } else {
            iterable = new LuceneIterable(reader, idField, field, mapper, norm);
          }
          String outFile = cmdLine.getValue(outputOpt).toString();
          log.info("Output File: " + outFile);

          VectorWriter vectorWriter;
          if (cmdLine.hasOption(outWriterOpt)) {
            String outWriter = cmdLine.getValue(outWriterOpt).toString();
            if (outWriter.equals("file")) {
              BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
              vectorWriter = new JWriterVectorWriter(writer);
            } else {
              vectorWriter = getSeqFileWriter(outFile);
            }
          } else {
            vectorWriter = getSeqFileWriter(outFile);
          }

          long numDocs = vectorWriter.write(iterable, maxDocs);
          vectorWriter.close();
          log.info("Wrote: " + numDocs + " vectors");

          String delimiter = cmdLine.hasOption(delimiterOpt) ? cmdLine.getValue(delimiterOpt).toString() : "\t";
          File dictOutFile = new File(cmdLine.getValue(dictOutOpt).toString());
          log.info("Dictionary Output file: " + dictOutFile);
          BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dictOutFile), Charset.forName("UTF8")));
          JWriterTermInfoWriter tiWriter = new JWriterTermInfoWriter(writer, delimiter, field);
          tiWriter.write(termInfo);
          tiWriter.close();
          writer.close();
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  private static VectorWriter getSeqFileWriter(String outFile) throws IOException {
    VectorWriter sfWriter;
    Path path = new Path(outFile);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    //TODO: Make this parameter driven
    SequenceFile.Writer seqWriter = SequenceFile.createWriter(fs, conf, path, LongWritable.class, SparseVector.class);

    sfWriter = new SequenceFileVectorWriter(seqWriter);
    return sfWriter;
  }


}
