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

package org.apache.mahout.classifier.bayes;

import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class WikipediaXmlSplitter {
  private WikipediaXmlSplitter() {
  }

  public static void main(String[] args) throws IOException, OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option dumpFileOpt = obuilder.withLongName("dumpFile").withRequired(true).withArgument(
            abuilder.withName("dumpFile").withMinimum(1).withMaximum(1).create()).
            withDescription("The path to the wikipedia dump file").withShortName("d").create();

    Option outputDirOpt = obuilder.withLongName("outputDir").withRequired(true).withArgument(
            abuilder.withName("outputDir").withMinimum(1).withMaximum(1).create()).
            withDescription("The output directory to place the splits in").withShortName("o").create();

    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withRequired(true).withArgument(
            abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).
            withDescription("The Size of the chunk, in megabytes").withShortName("c").create();
    Option numChunksOpt = obuilder.withLongName("numChunks").withRequired(false).withArgument(
            abuilder.withName("numChunks").withMinimum(1).withMaximum(1).create()).
            withDescription("The maximum number of chunks to create.  If specified, program will only create a subset of the chunks").withShortName("n").create();
    Group group = gbuilder.withName("Options").withOption(dumpFileOpt).withOption(outputDirOpt).withOption(chunkSizeOpt).withOption(numChunksOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);

    String dumpFilePath = (String) cmdLine.getValue(dumpFileOpt);
    String outputDirPath = (String) cmdLine.getValue(outputDirOpt);

    int chunkSize = 1024 * 1024 * Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));

    int numChunks = Integer.MAX_VALUE;
    if (cmdLine.hasOption(numChunksOpt)){
      numChunks = Integer.parseInt((String) cmdLine.getValue(numChunksOpt));
    }

    BufferedReader dumpReader = new BufferedReader(new InputStreamReader(
        new FileInputStream(dumpFilePath), "UTF-8"));

    File dir = new File(outputDirPath);
    dir.mkdirs();
    String header =
          "<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.mediawiki.org/xml/export-0.3/ http://www.mediawiki.org/xml/export-0.3.xsd\" version=\"0.3\" xml:lang=\"en\">\n"
        + "  <siteinfo>\n" + "<sitename>Wikipedia</sitename>\n"
        + "    <base>http://en.wikipedia.org/wiki/Main_Page</base>\n"
        + "    <generator>MediaWiki 1.13alpha</generator>\n"
        + "    <case>first-letter</case>\n"
        + "    <namespaces>\n"
        + "      <namespace key=\"-2\">Media</namespace>\n"
        + "      <namespace key=\"-1\">Special</namespace>\n"
        + "      <namespace key=\"0\" />\n"
        + "      <namespace key=\"1\">Talk</namespace>\n"
        + "      <namespace key=\"2\">User</namespace>\n"
        + "      <namespace key=\"3\">User talk</namespace>\n"
        + "      <namespace key=\"4\">Wikipedia</namespace>\n"
        + "      <namespace key=\"5\">Wikipedia talk</namespace>\n"
        + "      <namespace key=\"6\">Image</namespace>\n"
        + "      <namespace key=\"7\">Image talk</namespace>\n"
        + "      <namespace key=\"8\">MediaWiki</namespace>\n"
        + "      <namespace key=\"9\">MediaWiki talk</namespace>\n"
        + "      <namespace key=\"10\">Template</namespace>\n"
        + "      <namespace key=\"11\">Template talk</namespace>\n"
        + "      <namespace key=\"12\">Help</namespace>\n"
        + "      <namespace key=\"13\">Help talk</namespace>\n"
        + "      <namespace key=\"14\">Category</namespace>\n"
        + "      <namespace key=\"15\">Category talk</namespace>\n"
        + "      <namespace key=\"100\">Portal</namespace>\n"
        + "      <namespace key=\"101\">Portal talk</namespace>\n"
        + "    </namespaces>\n"
        + "  </siteinfo>\n";

    StringBuilder content = new StringBuilder();
    content.append(header);
    int filenumber = 0;
    NumberFormat decimalFormatter = new DecimalFormat("0000");
    String thisLine;
    while ((thisLine = dumpReader.readLine()) != null)
    {
      if(thisLine.trim().startsWith("<page>")){
        boolean end = false;
        while(thisLine.trim().startsWith("</page>") == false){
          content.append(thisLine).append('\n');
          if ((thisLine = dumpReader.readLine()) == null){
            end=true;
            break;
          }
        }
        content.append(thisLine).append('\n');

        if(content.length()>chunkSize || end){
          content.append("</mediawiki>");
          filenumber++;

          BufferedWriter chunkWriter = new BufferedWriter(new OutputStreamWriter(
              new FileOutputStream(dir.getPath()+"/chunk-"+ decimalFormatter.format(filenumber)+".xml"), "UTF-8"));

          chunkWriter.write(content.toString(), 0, content.length());
          chunkWriter.close();
          if (filenumber >= numChunks){
            break;
          }
          content = new StringBuilder();
          content.append(header);
        }
      }
    }

  }
}
