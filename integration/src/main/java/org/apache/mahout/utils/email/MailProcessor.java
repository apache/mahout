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

package org.apache.mahout.utils.email;

import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.utils.io.ChunkedWriter;
import org.apache.mahout.utils.io.ChunkedWrapper;
import org.apache.mahout.utils.io.IOWriterWrapper;
import org.apache.mahout.utils.io.WrappedWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Writer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts an mbox mail archive into a group of Hadoop Sequence Files with equal size. The archive may optionally be
 * gzipped or zipped. @see org.apache.mahout.text.SequenceFilesFromMailArchives
 */
public class MailProcessor {

  private static final Pattern MESSAGE_START = Pattern.compile("^From \\S+@\\S.*\\d{4}$", Pattern.CASE_INSENSITIVE);
  private static final Pattern MESSAGE_ID_PREFIX = Pattern.compile("^message-id: <(.*)>$", Pattern.CASE_INSENSITIVE);
  // regular expressions used to parse individual messages
  public static final Pattern SUBJECT_PREFIX = Pattern.compile("^subject: (.*)$", Pattern.CASE_INSENSITIVE);
  //we need to have at least one character
  public static final Pattern FROM_PREFIX = Pattern.compile("^from: (\\S.*)$", Pattern.CASE_INSENSITIVE);
  public static final Pattern REFS_PREFIX = Pattern.compile("^references: (.*)$", Pattern.CASE_INSENSITIVE);
  public static final Pattern TO_PREFIX = Pattern.compile("^to: (.*)$", Pattern.CASE_INSENSITIVE);

  private final String prefix;
  private final MailOptions options;
  private final WrappedWriter writer;

  private static final Logger log = LoggerFactory.getLogger(MailProcessor.class);

  /**
   * Creates a {@code MailProcessor} that does not write to sequence files, but to a single text file.
   * This constructor is for debugging and testing purposes.
   */
  public MailProcessor(MailOptions options, String prefix, Writer writer) {
    this.writer = new IOWriterWrapper(writer);
    this.options = options;
    this.prefix = prefix;
  }

  /**
   * This is the main constructor of {@code MailProcessor}.
   */
  public MailProcessor(MailOptions options, String prefix, ChunkedWriter writer) {
    this.writer = new ChunkedWrapper(writer);
    this.options = options;
    this.prefix = prefix;
  }

  /**
   * Parses one complete mail archive, writing output to the {@code writer} constructor parameter.
   * @param mboxFile  mail archive to parse
   * @return number of parsed mails
   * @throws IOException
   */
  public long parseMboxLineByLine(File mboxFile) throws IOException {
    long messageCount = 0;
    try {
      StringBuilder contents = new StringBuilder();
      // tmps used during mail message parsing
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
      for (String nextLine : new FileLineIterable(mboxFile, options.getCharset(), false)) {
        if (options.isStripQuotedText() && quotedTextPattern.matcher(nextLine).find()) {
          continue;
        }
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
            String key = generateKey(mboxFile, prefix, messageId);
            //if this ordering changes, then also change FromEmailToDictionaryMapper
            writeContent(options.getSeparator(), contents, body, patternResults);
            writer.write(key, contents.toString());
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
      // write the last message in the file if available
      if (messageId != null) {
        String key = generateKey(mboxFile, prefix, messageId);
        writeContent(options.getSeparator(), contents, body, patternResults);
        writer.write(key, contents.toString());
        contents.setLength(0); // reset the buffer
      }
    } catch (FileNotFoundException e) {
      // Skip file.
      log.warn("Unable to process non-existing file", e);
    }
    // TODO: report exceptions and continue;
    return messageCount;
  }

  protected static String generateKey(File mboxFile, String prefix, String messageId) {
    return prefix + File.separator + mboxFile.getName() + File.separator + messageId;
  }

  public String getPrefix() {
    return prefix;
  }

  public MailOptions getOptions() {
    return options;
  }

  private static void writeContent(String separator, StringBuilder contents, CharSequence body, String[] matches) {
    for (String match : matches) {
      if (match != null) {
        contents.append(match).append(separator);
      } else {
        contents.append(separator);
      }
    }
    contents.append('\n').append(body);
  }
}
