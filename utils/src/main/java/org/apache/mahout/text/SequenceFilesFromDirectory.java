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

package org.apache.mahout.text;

import java.io.Closeable;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;

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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a directory of text documents into SequenceFiles of Specified chunkSize. This class takes in a
 * parent directory containing sub folders of text documents and recursively reads the files and creates the
 * {@link SequenceFile}s of docid => content. The docid is set as the relative path of the document from the
 * parent directory prepended with a specified prefix. You can also specify the input encoding of the text
 * files. The content of the output SequenceFiles are encoded as UTF-8 text.
 * 
 * 
 */
public final class SequenceFilesFromDirectory {
  
  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromDirectory.class);
  
  private static ChunkedWriter createNewChunkedWriter(int chunkSizeInMB, String outputDir) throws IOException {
    return new ChunkedWriter(chunkSizeInMB, outputDir);
  }
  
  public void createSequenceFiles(File parentDir,
                                  String outputDir,
                                  String prefix,
                                  int chunkSizeInMB,
                                  Charset charset) throws IOException {
    ChunkedWriter writer = createNewChunkedWriter(chunkSizeInMB, outputDir);
    parentDir.listFiles(new PrefixAdditionFilter(prefix, writer, charset));
    writer.close();
  }
  
  public static class ChunkedWriter implements Closeable {
    private final int maxChunkSizeInBytes;
    private final String outputDir;
    private SequenceFile.Writer writer;
    private int currentChunkID;
    private int currentChunkSize;
    private final Configuration conf = new Configuration();
    private final FileSystem fs;
    
    public ChunkedWriter(int chunkSizeInMB, String outputDir) throws IOException {
      if (chunkSizeInMB > 1984) {
        chunkSizeInMB = 1984;
      }
      maxChunkSizeInBytes = chunkSizeInMB * 1024 * 1024;
      this.outputDir = outputDir;
      fs = FileSystem.get(conf);
      currentChunkID = 0;
      writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID), Text.class, Text.class);
    }
    
    private Path getPath(int chunkID) {
      return new Path(outputDir + "/chunk-" + chunkID);
    }
    
    public void write(String key, String value) throws IOException {
      if (currentChunkSize > maxChunkSizeInBytes) {
        writer.close();
        writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID++), Text.class, Text.class);
        currentChunkSize = 0;
        
      }
      
      Text keyT = new Text(key);
      Text valueT = new Text(value);
      currentChunkSize += keyT.getBytes().length + valueT.getBytes().length; // Overhead
      writer.append(keyT, valueT);
    }
    
    @Override
    public void close() throws IOException {
      writer.close();
    }
  }
  
  public class PrefixAdditionFilter implements FileFilter {
    private final String prefix;
    private final ChunkedWriter writer;
    private final Charset charset;
    
    public PrefixAdditionFilter(String prefix, ChunkedWriter writer, Charset charset) {
      this.prefix = prefix;
      this.writer = writer;
      this.charset = charset;
    }
    
    @Override
    public boolean accept(File current) {
      if (current.isDirectory()) {
        current.listFiles(new PrefixAdditionFilter(prefix + File.separator + current.getName(), writer,
            charset));
      } else {
        try {
          StringBuilder file = new StringBuilder();
          for (String aFit : new FileLineIterable(current, charset, false)) {
            file.append(aFit).append('\n');
          }
          writer.write(prefix + File.separator + current.getName(), file.toString());
          
        } catch (FileNotFoundException e) {
          // Skip file.
        } catch (IOException e) {
          // TODO: report exceptions and continue;
          throw new IllegalStateException(e);
        }
      }
      return false;
    }
    
  }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option parentOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The input dir containing the documents").withShortName("i").create();
    
    Option outputDirOpt = obuilder.withLongName("output").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "The output directory").withShortName("o").create();
    
    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withArgument(
      abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "The chunkSize in MegaBytes. Defaults to 64").withShortName("chunk").create();
    
    Option keyPrefixOpt = obuilder.withLongName("keyPrefix").withArgument(
      abuilder.withName("keyPrefix").withMinimum(1).withMaximum(1).create()).withDescription(
      "The prefix to be prepended to the key").withShortName("prefix").create();
    
    Option charsetOpt = obuilder.withLongName("charset").withRequired(true).withArgument(
      abuilder.withName("charset").withMinimum(1).withMaximum(1).create()).withDescription(
      "The name of the character encoding of the input files").withShortName("c").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(keyPrefixOpt).withOption(chunkSizeOpt).withOption(
      charsetOpt).withOption(outputDirOpt).withOption(helpOpt).withOption(parentOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      File parentDir = new File((String) cmdLine.getValue(parentOpt));
      String outputDir = (String) cmdLine.getValue(outputDirOpt);
      
      int chunkSize = 64;
      if (cmdLine.hasOption(chunkSizeOpt)) {
        chunkSize = Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));
      }
      
      String prefix = "";
      if (cmdLine.hasOption(keyPrefixOpt)) {
        prefix = (String) cmdLine.getValue(keyPrefixOpt);
      }
      Charset charset = Charset.forName((String) cmdLine.getValue(charsetOpt));
      SequenceFilesFromDirectory dir = new SequenceFilesFromDirectory();
      
      dir.createSequenceFiles(parentDir, outputDir, prefix, chunkSize, charset);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
}
