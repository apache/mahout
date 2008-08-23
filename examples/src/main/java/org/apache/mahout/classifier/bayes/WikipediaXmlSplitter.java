package org.apache.mahout.classifier.bayes;

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
import org.apache.commons.cli.PosixParser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;

public class WikipediaXmlSplitter {

  @SuppressWarnings("static-access")
  public static void main(String[] args) throws Exception {
    Options options = new Options();
    Option dumpFileOpt = OptionBuilder.withLongOpt("dumpfile").isRequired().hasArg().withDescription("The path to the wikipedia dump file").create("d");
    options.addOption(dumpFileOpt);
    Option outputDirOpt = OptionBuilder.withLongOpt("outputDir").isRequired().hasArg().withDescription("The output directory to place the splits in").create("o");
    options.addOption(outputDirOpt);
    Option chunkSizeOpt = OptionBuilder.withLongOpt("chunkSize").isRequired().hasArg().withDescription("the Size of chunk in Megabytes").create("c");
    options.addOption(chunkSizeOpt);
    CommandLine cmdLine;
    try {
      PosixParser parser = new PosixParser();
      cmdLine = parser.parse(options, args);

      String dumpFilePath = cmdLine.getOptionValue(dumpFileOpt.getOpt());
      String outputDirPath = cmdLine.getOptionValue(outputDirOpt.getOpt());
      int chunkSize = 1024 * 1024 * Integer.parseInt(cmdLine.getOptionValue(chunkSizeOpt.getOpt()));
      
      BufferedReader dumpReader = new BufferedReader(new InputStreamReader(
          new FileInputStream(dumpFilePath), "UTF-8"));

      File dir = new File(outputDirPath);
      dir.getPath();
      

      String header = ""
          + "<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.3/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.mediawiki.org/xml/export-0.3/ http://www.mediawiki.org/xml/export-0.3.xsd\" version=\"0.3\" xml:lang=\"en\">\n"
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
      String thisLine;
      StringBuilder content = new StringBuilder();
      content.append(header);
      Integer filenumber = new Integer(0);
      DecimalFormat decimalFormatter = new DecimalFormat("0000");
      while ((thisLine = dumpReader.readLine()) != null) 
      {
        boolean end = false;
        if(thisLine.trim().startsWith("<page>")){
          while(thisLine.trim().startsWith("</page>")==false){
            content.append(thisLine).append("\n"); 
            if ((thisLine = dumpReader.readLine()) == null){
              end=true;
              break;
            }
          }
          content.append(thisLine).append("\n");
          
          if(content.length()>chunkSize || end){
            content.append("</mediawiki>");
            filenumber++;
            
            BufferedWriter chunkWriter = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dir.getPath()+"/chunk-"+ decimalFormatter.format(filenumber)+".xml"), "UTF-8"));
            
            chunkWriter.write(content.toString(), 0, content.length());
            chunkWriter.close();
            
            content = new StringBuilder();
            
            content.append(header);
            
          }
        }
      } 

    } catch (Exception exp) {
      exp.printStackTrace(System.err);
    }
  }
}
