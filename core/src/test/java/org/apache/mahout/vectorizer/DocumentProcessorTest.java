package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.StringTuple;
import org.junit.Test;

import java.util.Arrays;

/**
 * Tests tokenizing of <Text documentId, Text text> {@link SequenceFile}s by the {@link DocumentProcessor} into
 * <Text documentId, StringTuple tokens> sequence files
 */
public class DocumentProcessorTest extends MahoutTestCase {

  @Test
  public void testTokenizeDocuments() throws Exception {
    Configuration configuration = new Configuration();
    FileSystem fs = FileSystem.get(configuration);
    Path input = new Path(getTestTempDirPath(), "inputDir");
    Path output = new Path(getTestTempDirPath(), "outputDir");

    String documentId1 = "123";
    String text1 = "A test for the document processor";

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, configuration, input, Text.class, Text.class);
    writer.append(new Text(documentId1), new Text(text1));
    String documentId2 = "456";
    String text2 = "and another one";
    writer.append(new Text(documentId2), new Text(text2));
    writer.close();

    DocumentProcessor.tokenizeDocuments(input, DefaultAnalyzer.class, output);

    FileStatus[] statuses = fs.listStatus(output);
    assertEquals(1, statuses.length);
    Path filePath = statuses[0].getPath();
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, filePath, configuration);
    Text key = reader.getKeyClass().asSubclass(Text.class).newInstance();
    StringTuple value = reader.getValueClass().asSubclass(StringTuple.class).newInstance();

    reader.next(key, value);
    assertEquals(documentId1, key.toString());
    assertEquals(Arrays.asList("test", "document", "processor"), value.getEntries());
    reader.next(key, value);
    assertEquals(documentId2, key.toString());
    assertEquals(Arrays.asList("another", "one"), value.getEntries());
  }
}
