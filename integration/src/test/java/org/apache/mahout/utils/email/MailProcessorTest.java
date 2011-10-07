package org.apache.mahout.utils.email;


import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.io.File;
import java.io.StringWriter;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.regex.Pattern;

/**
 *
 *
 **/
public class MailProcessorTest extends MahoutTestCase {
  @Test
  public void testLabel() throws Exception {
    StringWriter writer = new StringWriter();
    MailOptions options = new MailOptions();
    options.separator = ":::";
    options.charset = Charset.forName("UTF-8");
    options.patternsToMatch = new Pattern[]{MailProcessor.FROM_PREFIX, MailProcessor.SUBJECT_PREFIX, MailProcessor.TO_PREFIX};
    options.input = new File(System.getProperty("user.dir"));
    MailProcessor proc = new MailProcessor(options, "", writer);
    URL url = MailProcessorTest.class.getClassLoader().getResource("test.mbox");
    File file = new File(url.toURI());
    //System.out.println(file);
    long count = proc.parseMboxLineByLine(file);
    assertEquals(7, count);
    System.out.println(writer.getBuffer());
    //TODO

  }



}
