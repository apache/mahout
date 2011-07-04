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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.io.Closeables;
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
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a directory of gzipped mail archives into SequenceFiles of specified chunkSize.
 * This class is similar to {@link SequenceFilesFromDirectory} except it uses block-compressed
 * {@link SequenceFile}s and parses out the subject and body text of each mail message into
 * a separate key/value pair.
 */
public final class SequenceFilesFromMailArchives {

  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromMailArchives.class);
  
  private static ChunkedWriter createNewChunkedWriter(int chunkSizeInMB, String outputDir) throws IOException {
    return new ChunkedWriter(chunkSizeInMB, outputDir);
  }
  
  public void createSequenceFiles(File parentDir,
                                  String outputDir,
                                  String prefix,
                                  int chunkSizeInMB,
                                  Charset charset) throws IOException {
    ChunkedWriter writer = createNewChunkedWriter(chunkSizeInMB, outputDir);
    try {
      PrefixAdditionFilter filter = new PrefixAdditionFilter(prefix, writer, charset);
      parentDir.listFiles(filter);
      log.info("Parsed "+filter.getMessageCount()+" messages from "+parentDir.getAbsolutePath());
    } finally {
      Closeables.closeQuietly(writer);
    }
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
      
      writer = SequenceFile.createWriter(fs, conf, getPath(currentChunkID), Text.class, Text.class, SequenceFile.CompressionType.BLOCK);      
    }
    
    private Path getPath(int chunkID) {
      return new Path(outputDir + "/chunk-" + chunkID);
    }
    
    public void write(String key, String value) throws IOException {
      if (currentChunkSize > maxChunkSizeInBytes) {
        Closeables.closeQuietly(writer);
        log.info("Chunk size ("+currentChunkSize+") reached MAX; creating new chunk "+(currentChunkID+1));
        writer = SequenceFile.createWriter(fs, conf, getPath(currentChunkID++), Text.class, Text.class, SequenceFile.CompressionType.BLOCK);
        currentChunkSize = 0;        
      }
      
      Text keyT = new Text(key);
      Text valueT = new Text(value);
      currentChunkSize += keyT.getBytes().length + valueT.getBytes().length; // Overhead
      writer.append(keyT, valueT);
    }
    
    @Override
    public void close() throws IOException {
      Closeables.closeQuietly(writer);
    }
  }
  
  // regular expressions used to parse individual messages
  private static final Pattern MESSAGE_START = 
    Pattern.compile("^From \\S+@\\S.*\\d{4}$", Pattern.CASE_INSENSITIVE);
  private static final Pattern MESSAGE_ID_PREFIX = 
    Pattern.compile("^message-id: <(.*)>$", Pattern.CASE_INSENSITIVE);
  private static final Pattern SUBJECT_PREFIX = 
    Pattern.compile("^subject: (.*)$", Pattern.CASE_INSENSITIVE);  
  
  public class PrefixAdditionFilter implements FileFilter {
    private final String prefix;
    private final ChunkedWriter writer;
    private final Charset charset;
    private final StringBuilder file;
    private int messageCount;
    
    public PrefixAdditionFilter(String prefix, ChunkedWriter writer, Charset charset) {
      this.prefix = prefix;
      this.writer = writer;
      this.charset = charset;
      this.file = new StringBuilder();
      this.messageCount = 0;
    }
    
    public int getMessageCount() {
      return messageCount;
    }
    
    @Override
    public boolean accept(File current) {
      if (current.isDirectory()) {
        log.info("At "+current.getAbsolutePath());
        PrefixAdditionFilter nested = 
          new PrefixAdditionFilter(prefix + File.separator + current.getName(), writer, charset);
        current.listFiles(nested);
        int dirCount = nested.getMessageCount();
        log.info("Parsed "+dirCount+" messages from directory "+current.getAbsolutePath());
        messageCount += dirCount;
      } else {
        try {
          parseFileLineByLine(current);
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
      }
      return false;
    }
    
    // extracts mail subject and body text from 0 or more mail messages
    // embedded in the supplied file using simple pattern matching
    private void parseFileLineByLine(File current) throws IOException {
      try {
        file.setLength(0); // reset the buffer
        
        // tmps used during mail message parsing
        String messageId = null;
        boolean inBody = false;
        Matcher subjectMatcher = SUBJECT_PREFIX.matcher("");
        Matcher messageIdMatcher = MESSAGE_ID_PREFIX.matcher("");
        Matcher messageBoundaryMatcher = MESSAGE_START.matcher("");
        
        for (String nextLine : new FileLineIterable(current, charset, false)) {

          // subject may come before message ID
          subjectMatcher.reset(nextLine);
          if (subjectMatcher.matches()) {
            file.append(subjectMatcher.group(1)).append('\n');
          }
          
          // only start appending body content after we've seen a message ID
          if (messageId != null) {            
            // first, see if we hit the end of the message
            messageBoundaryMatcher.reset(nextLine);              
            if (messageBoundaryMatcher.matches()) {
                // done parsing this message ... write it out
                String key = prefix + File.separator + current.getName() + File.separator + messageId;
                writer.write(key, file.toString());
                file.setLength(0); // reset the buffer
                messageId = null;
                inBody = false;
            } else {
              if (inBody) {
                if (nextLine.length() > 0) {
                  file.append(nextLine).append('\n');
                }
              } else {
                // first empty line we see after reading the message Id
                // indicates that we are in the body ...
                inBody = nextLine.length() == 0;
              }
            }
          } else {
            if (nextLine.length() > 14) {
              messageIdMatcher.reset(nextLine);
              if (messageIdMatcher.matches()) {
                messageId = messageIdMatcher.group(1);
                ++messageCount;
              }
            }
          }
        }

        // write the last message in the file if available
        if (messageId != null) {
          String key = prefix + File.separator + current.getName() + File.separator + messageId;
          writer.write(key, file.toString());
          file.setLength(0); // reset the buffer
        }
      } catch (FileNotFoundException e) {
        // Skip file.
      }
      // TODO: report exceptions and continue;

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
      parser.setHelpOption(helpOpt);
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
      SequenceFilesFromMailArchives dir = new SequenceFilesFromMailArchives();
      
      dir.createSequenceFiles(parentDir, outputDir, prefix, chunkSize, charset);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
}
