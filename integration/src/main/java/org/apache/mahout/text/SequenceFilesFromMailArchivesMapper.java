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

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.utils.email.MailOptions;
import org.apache.mahout.utils.email.MailProcessor;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.mahout.text.SequenceFilesFromMailArchives.BODY_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.BODY_SEPARATOR_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.CHARSET_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.CHUNK_SIZE_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.FROM_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.KEY_PREFIX_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.QUOTED_REGEX_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.REFERENCES_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.SEPARATOR_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.STRIP_QUOTED_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.SUBJECT_OPTION;
import static org.apache.mahout.text.SequenceFilesFromMailArchives.TO_OPTION;

/**
 * Map Class for the SequenceFilesFromMailArchives job
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

    Configuration configuration = context.getConfiguration();

    // absorb all of the options into the MailOptions object
    this.options = new MailOptions();

    options.setPrefix(configuration.get(KEY_PREFIX_OPTION[1], ""));

    if (!configuration.get(CHUNK_SIZE_OPTION[0], "").equals("")) {
      options.setChunkSize(configuration.getInt(CHUNK_SIZE_OPTION[0], 64));
    }

    if (!configuration.get(CHARSET_OPTION[0], "").equals("")) {
      Charset charset = Charset.forName(configuration.get(CHARSET_OPTION[0], "UTF-8"));
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
    Map<String, Integer> patternOrder = Maps.newHashMap();
    int order = 0;
    if (!configuration.get(FROM_OPTION[1], "").equals("")) {
      patterns.add(MailProcessor.FROM_PREFIX);
      patternOrder.put(MailOptions.FROM, order++);
    }

    if (!configuration.get(TO_OPTION[1], "").equals("")) {
      patterns.add(MailProcessor.TO_PREFIX);
      patternOrder.put(MailOptions.TO, order++);
    }

    if (!configuration.get(REFERENCES_OPTION[1], "").equals("")) {
      patterns.add(MailProcessor.REFS_PREFIX);
      patternOrder.put(MailOptions.REFS, order++);
    }

    if (!configuration.get(SUBJECT_OPTION[1], "").equals("")) {
      patterns.add(MailProcessor.SUBJECT_PREFIX);
      patternOrder.put(MailOptions.SUBJECT, order += 1);
    }

    options.setStripQuotedText(configuration.getBoolean(STRIP_QUOTED_OPTION[1], false));

    options.setPatternsToMatch(patterns.toArray(new Pattern[patterns.size()]));
    options.setPatternOrder(patternOrder);

    options.setIncludeBody(configuration.getBoolean(BODY_OPTION[1], false));

    options.setSeparator("\n");
    if (!configuration.get(SEPARATOR_OPTION[1], "").equals("")) {
      options.setSeparator(configuration.get(SEPARATOR_OPTION[1], ""));
    }
    if (!configuration.get(BODY_SEPARATOR_OPTION[1], "").equals("")) {
      options.setBodySeparator(configuration.get(BODY_SEPARATOR_OPTION[1], ""));
    }
    if (!configuration.get(QUOTED_REGEX_OPTION[1], "").equals("")) {
      options.setQuotedTextPattern(Pattern.compile(configuration.get(QUOTED_REGEX_OPTION[1], "")));
    }

  }

  public long parseMailboxLineByLine(String filename, InputStream mailBoxInputStream, Context context)
    throws IOException, InterruptedException {
    long messageCount = 0;
    try {
      StringBuilder contents = new StringBuilder();
      StringBuilder body = new StringBuilder();
      Matcher messageIdMatcher = MESSAGE_ID_PREFIX.matcher("");
      Matcher messageBoundaryMatcher = MESSAGE_START.matcher("");
      String[] patternResults = new String[options.getPatternsToMatch().length];
      Matcher[] matches = new Matcher[options.getPatternsToMatch().length];
      for (int i = 0; i < matches.length; i++) {
        matches[i] = options.getPatternsToMatch()[i].matcher("");
      }

      String messageId = null;
      boolean inBody = false;
      Pattern quotedTextPattern = options.getQuotedTextPattern();

      for (String nextLine : new FileLineIterable(mailBoxInputStream, options.getCharset(), false, filename)) {
        if (!options.isStripQuotedText() || !quotedTextPattern.matcher(nextLine).find()) {
          for (int i = 0; i < matches.length; i++) {
            Matcher matcher = matches[i];
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
    return messageCount;
  }

  protected static String generateKey(String mboxFilename, String prefix, String messageId) {
    return Joiner.on(Path.SEPARATOR).join(Lists.newArrayList(prefix, mboxFilename, messageId).iterator());
  }

  private static void writeContent(String separator, StringBuilder contents, CharSequence body, String[] matches) {
    String matchesString = Joiner.on(separator).useForNull("").join(Arrays.asList(matches).iterator());
    contents.append(matchesString).append(separator).append(body);
  }

  public void map(IntWritable key, BytesWritable value, Context context)
    throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();
    Path filePath = ((CombineFileSplit) context.getInputSplit()).getPath(key.get());
    String relativeFilePath = HadoopUtil.calcRelativeFilePath(configuration, filePath);
    ByteArrayInputStream is = new ByteArrayInputStream(value.getBytes());
    parseMailboxLineByLine(relativeFilePath, is, context);
  }
}
