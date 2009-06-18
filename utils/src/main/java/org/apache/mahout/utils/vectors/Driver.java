package org.apache.mahout.utils.vectors;
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
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.lucene.CachedTermInfo;
import org.apache.mahout.utils.vectors.lucene.LuceneIteratable;
import org.apache.mahout.utils.vectors.lucene.TFDFMapper;
import org.apache.mahout.utils.vectors.lucene.VectorMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.Iterator;


/**
 *
 *
 **/
public class Driver {
  private transient static Logger log = LoggerFactory.getLogger(Driver.class);
  //TODO: This assumes LuceneIterable, make it generic.
  
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
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(idFieldOpt).withOption(outputOpt).withOption(delimiterOpt)
            .withOption(helpOpt).withOption(fieldOpt).withOption(maxOpt).withOption(dictOutOpt).withOption(powerOpt)
            .withOption(weightOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        printHelp(group);
        return;
      }
      //Springify all this
      if (cmdLine.hasOption(inputOpt)) {//Lucene case
        File file = new File(cmdLine.getValue(inputOpt).toString());
        if (file.exists() && file.isDirectory()) {
          int maxDocs = Integer.MAX_VALUE;
          if (cmdLine.hasOption(maxOpt)) {
            maxDocs = Integer.parseInt(cmdLine.getValue(maxOpt).toString());
          }
          if (maxDocs < 0) {
            throw new IllegalArgumentException("maxDocs must be >= 0");
          }
          Directory dir = FSDirectory.open(file);
          IndexReader reader = IndexReader.open(dir, true);
          Weight weight = null;
          if(cmdLine.hasOption(weightOpt)) {
            String wString = cmdLine.getValue(weightOpt).toString();
            if(wString.equalsIgnoreCase("tf")) {
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
          TermInfo termInfo = new CachedTermInfo(reader, field, 1, 99);
          VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
          LuceneIteratable iteratable = null;
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
          if (cmdLine.hasOption(idFieldOpt)){
            idField = cmdLine.getValue(idFieldOpt).toString();
          }
          if (norm == LuceneIteratable.NO_NORMALIZING) {
            iteratable = new LuceneIteratable(reader, idField, field, mapper, LuceneIteratable.NO_NORMALIZING);
          } else {
            iteratable = new LuceneIteratable(reader, idField, field, mapper, norm);
          }
          File outFile = new File(cmdLine.getValue(outputOpt).toString());
          log.info("Output File: " + outFile);
          BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
          int i = 0;
          for (Vector vector : iteratable) {
            if (i >= maxDocs){
              break;
            }
            writer.write(vector.asFormatString());
            writer.write("\n");
            if (i % 500 == 0) {
              log.info("i = " + i);
            }
            i++;
          }
          log.info("Wrote " + i + " vectors");
          writer.flush();
          writer.close();
          // TODO: replace with aa codec
          File dictOutFile = new File(cmdLine.getValue(dictOutOpt).toString());
          log.info("Dictionary Output file: " + dictOutFile);
          writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dictOutFile), Charset.forName("UTF8")));
          Iterator<TermEntry> entIter = termInfo.getAllEntries();
          String delimiter = cmdLine.hasOption(delimiterOpt) ? cmdLine.getValue(delimiterOpt).toString() : "\t";
          writer.write("input");
          writer.write(delimiter);
          writer.write(file.getAbsolutePath());
          writer.write("\n");
          writer.write("field");
          writer.write(delimiter);
          writer.write(field);
          writer.write("\n");
          writer.write("num.terms");
          writer.write(delimiter);
          writer.write(String.valueOf(termInfo.totalTerms(field)));
          writer.write("\n");
          writer.write("#term" + delimiter + "doc freq" + delimiter + "idx");
          writer.write("\n");
          while (entIter.hasNext()) {
            TermEntry entry = entIter.next();
            writer.write(entry.term);
            writer.write(delimiter);
            writer.write(String.valueOf(entry.docFreq));
            writer.write(delimiter);
            writer.write(String.valueOf(entry.termIdx));
            writer.write("\n");
          }
          writer.flush();
          writer.close();
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      printHelp(group);
    }
  }

  private static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
}
