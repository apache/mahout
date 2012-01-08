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

import com.google.common.base.Charsets;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.io.File;
import java.io.StringWriter;
import java.net.URL;
import java.util.regex.Pattern;

public final class MailProcessorTest extends MahoutTestCase {

  @Test
  public void testLabel() throws Exception {
    StringWriter writer = new StringWriter();
    MailOptions options = new MailOptions();
    options.setSeparator(":::");
    options.setCharset(Charsets.UTF_8);
        options.setPatternsToMatch(new Pattern[]{
        MailProcessor.FROM_PREFIX, MailProcessor.SUBJECT_PREFIX, MailProcessor.TO_PREFIX});
    options.setInput(new File(System.getProperty("user.dir")));
    MailProcessor proc = new MailProcessor(options, "", writer);
    URL url = MailProcessorTest.class.getClassLoader().getResource("test.mbox");
    File file = new File(url.toURI());
    long count = proc.parseMboxLineByLine(file);
    assertEquals(7, count);
  }

  @Test
  public void testStripQuoted() throws Exception {
    StringWriter writer = new StringWriter();
    MailOptions options = new MailOptions();
    options.setSeparator(":::");
    options.setCharset(Charsets.UTF_8);
        options.setPatternsToMatch(new Pattern[]{
        MailProcessor.SUBJECT_PREFIX});
    options.setInput(new File(System.getProperty("user.dir")));
    options.setIncludeBody(true);
    MailProcessor proc = new MailProcessor(options, "", writer);
    URL url = MailProcessorTest.class.getClassLoader().getResource("test.mbox");
    File file = new File(url.toURI());
    long count = proc.parseMboxLineByLine(file);
    assertEquals(7, count);
    assertTrue(writer.getBuffer().toString().contains("> Cocoon Cron Block Configurable Clustering"));
    writer = new StringWriter();
    proc = new MailProcessor(options, "", writer);
    options.setStripQuotedText(true);
    count = proc.parseMboxLineByLine(file);
    assertEquals(7, count);
    assertFalse(writer.getBuffer().toString().contains("> Cocoon Cron Block Configurable Clustering"));

  }

}
