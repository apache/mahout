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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.utils.email.MailOptions;
import org.apache.mahout.utils.email.MailProcessor;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;

/**
 * 
 * Map Class for the SequenceFilesFromMailArchives job
 * 
 */
public class SequenceFilesFromMailArchivesMapper extends Mapper<IntWritable, BytesWritable, Text, Text> {
  
  private Text outKey = new Text();
  private Text outValue = new Text();
  
  private static final Pattern MESSAGE_START = Pattern.compile(
      "^From \\S+@\\S.*\\d{4}$", Pattern.CASE_INSENSITIVE);
  private static final Pattern MESSAGE_ID_PREFIX = Pattern.compile(
      "^message-id: <(.*)>$", Pattern.CASE_INSENSITIVE);

  private MailOptions options;
  
  @Override
  public void setup(Context context) throws IOException, InterruptedException {

    Configuration conf = context.getConfiguration();
    // absorb all of the options into the MailOptions object
    
    this.options = new MailOptions();

    options.setPrefix(conf.get("prefix", ""));
    
    if (!conf.get("chunkSize", "").equals("")) {
      options.setChunkSize(conf.getInt("chunkSize", 64));
    }
    
    if (!conf.get("charset", "").equals("")) {
      Charset charset = Charset.forName(conf.get("charset", "UTF-8"));
      options.setCharset(charset);
    } else {
      Charset charset = Charset.forName("UTF-8");
      options.setCharset(charset);
    }
    
    List<Pattern> patterns = Lists.newArrayListWithCapacity(5);
    // patternOrder is used downstream so that we can know what order the
    // text is in instead
    // of encoding it in the string, which
    // would require more processing later to remove it pre feature
    // selection.
    Map<String,Integer> patternOrder = Maps.newHashMap();
    int order = 0;
    
    if (!conf.get("fromOpt", "").equals("")) {
      patterns.add(MailProcessor.FROM_PREFIX);
      patternOrder.put(MailOptions.FROM, order++);
    }

    if (!conf.get("toOpt", "").equals("")) {
      patterns.add(MailProcessor.TO_PREFIX);
      patternOrder.put(MailOptions.TO, order++);
    }

    if (!conf.get("refsOpt", "").equals("")) {
      patterns.add(MailProcessor.REFS_PREFIX);
      patternOrder.put(MailOptions.REFS, order++);
    }
    
    if (!conf.get("subjectOpt", "").equals("")) {
      patterns.add(MailProcessor.SUBJECT_PREFIX);
      patternOrder.put(MailOptions.SUBJECT, order++);
    }
    
    options.setStripQuotedText(conf.getBoolean("quotedOpt", false));
    
    options.setPatternsToMatch(patterns.toArray(new Pattern[patterns.size()]));
    options.setPatternOrder(patternOrder);
    
    options.setIncludeBody(conf.getBoolean("bodyOpt", false));
    
    options.setSeparator("\n");
    if (!conf.get("separatorOpt", "").equals("")) {
      options.setSeparator(conf.get("separatorOpt", ""));
    }
    if (!conf.get("bodySeparatorOpt", "").equals("")) {
      options.setBodySeparator(conf.get("bodySeparatorOpt", ""));
    }
    if (!conf.get("quotedRegexOpt", "").equals("")) {
      options.setQuotedTextPattern(Pattern.compile(conf.get("quotedRegexOpt", "")));
    }

  }
  
  public long parseMboxLineByLine(String filename, InputStream mboxInputStream, Context context)
    throws IOException, InterruptedException {
    long messageCount = 0;
    try {
      StringBuilder contents = new StringBuilder();
      StringBuilder body = new StringBuilder();
      Matcher messageIdMatcher = MESSAGE_ID_PREFIX.matcher("");
      Matcher messageBoundaryMatcher = MESSAGE_START.matcher("");
      String[] patternResults = new String[options.getPatternsToMatch().length];
      Matcher[] matchers = new Matcher[options.getPatternsToMatch().length];
      for (int i = 0; i < matchers.length; i++) {
        matchers[i] = options.getPatternsToMatch()[i].matcher("");
      }
      
      String messageId = null;
      boolean inBody = false;
      Pattern quotedTextPattern = options.getQuotedTextPattern();
      
      for (String nextLine : new FileLineIterable(mboxInputStream, options.getCharset(), false, filename)) {
        if (!options.isStripQuotedText() || !quotedTextPattern.matcher(nextLine).find()) {
          for (int i = 0; i < matchers.length; i++) {
            Matcher matcher = matchers[i];
            matcher.reset(nextLine);
            if (matcher.matches()) {
              patternResults[i] = matcher.group(1);
            }
          }

          // only start appending body content after we've seen a message ID
          if (messageId != null) {
            // first, see if we hit the end of the message
            messageBoundaryMatcher.reset(nextLine);
            if (messageBoundaryMatcher.matches()) {
              // done parsing this message ... write it out
              String key = generateKey(filename, options.getPrefix(), messageId);
              // if this ordering changes, then also change
              // FromEmailToDictionaryMapper
              writeContent(options.getSeparator(), contents, body, patternResults);

              this.outKey.set(key);
              this.outValue.set(contents.toString());
              context.write(this.outKey, this.outValue);
              contents.setLength(0); // reset the buffer
              body.setLength(0);
              messageId = null;
              inBody = false;
            } else {
              if (inBody && options.isIncludeBody()) {
                if (!nextLine.isEmpty()) {
                  body.append(nextLine).append(options.getBodySeparator());
                }
              } else {
                // first empty line we see after reading the message Id
                // indicates that we are in the body ...
                inBody = nextLine.isEmpty();
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
      }
      // write the last message in the file if available
      if (messageId != null) {
        String key = generateKey(filename, options.getPrefix(), messageId);
        writeContent(options.getSeparator(), contents, body, patternResults);
        
        this.outKey.set(key);
        this.outValue.set(contents.toString());
        context.write(this.outKey, this.outValue);
        contents.setLength(0); // reset the buffer
      }
    } catch (FileNotFoundException ignored) {

    }
    // TODO: report exceptions and continue;
    return messageCount;
  }
  
  protected static String generateKey(String mboxFilename, String prefix, String messageId) {
    return prefix + File.separator + mboxFilename + File.separator + messageId;
  }
  
  private static void writeContent(String separator, StringBuilder contents, CharSequence body, String[] matches) {
    for (String match : matches) {
      if (match != null) {
        contents.append(match).append(separator);
      } else {
        contents.append("").append(separator);
      }
    }
    contents.append(body);
  }
  
  public static String calcRelativeFilePath(Configuration conf, Path filePath) throws IOException {
    FileSystem fs = filePath.getFileSystem(conf);
    FileStatus fst = fs.getFileStatus(filePath);
    String currentPath = fst.getPath().toString().replaceFirst("file:", "");

    String basePath = conf.get("baseinputpath");
    if (!basePath.endsWith("/")) {
      basePath += "/";
    }
    basePath = basePath.replaceFirst("file:", "");
    String[] parts = currentPath.split(basePath);

    String hdfsStuffRemoved = currentPath; // default value
    if (parts.length == 2) {
      hdfsStuffRemoved = parts[1];
    } else if (parts.length == 1) {
      hdfsStuffRemoved = parts[0];
    }
    return hdfsStuffRemoved;
  }

  public void map(IntWritable key, BytesWritable value, Context context)
    throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();
    Path filePath = ((CombineFileSplit) context.getInputSplit()).getPath(key.get());
    String relativeFilePath = calcRelativeFilePath(configuration, filePath);
    ByteArrayInputStream is = new ByteArrayInputStream(value.getBytes());
    parseMboxLineByLine(relativeFilePath, is, context);
  }
}
