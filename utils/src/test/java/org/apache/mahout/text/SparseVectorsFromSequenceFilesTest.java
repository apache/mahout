package org.apache.mahout.text;

import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.utils.MahoutTestCase;
import org.apache.mahout.utils.vectors.text.DictionaryVectorizerTest;
import org.apache.mahout.utils.vectors.text.RandomDocumentGenerator;
import org.junit.Before;
import org.junit.Test;


public class SparseVectorsFromSequenceFilesTest extends MahoutTestCase {
  private static final int NUM_DOCS = 100;
  
  private Configuration conf;
  private FileSystem fs;
  private Path inputPath;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = new Configuration();
    fs = FileSystem.get(conf);

    inputPath = getTestTempFilePath("documents/docs.file");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    RandomDocumentGenerator gen = new RandomDocumentGenerator();
    
    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(gen.getRandomDocument()));
    }
    writer.close();
  }
  
  
  @Test
  public void testCreateTermFrequencyVectors() throws Exception {
    runTest(false, false);
  }

  @Test
  public void testCreateTermFrequencyVectorsNam() throws Exception {
    runTest(false, true);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeq() throws Exception {
    runTest(true, false);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeqNam() throws Exception {
    runTest(true, true);
  }
  
  protected void runTest(boolean sequential, boolean named) throws Exception {
    Path outputPath = getTestTempFilePath("output");

    
    List<String> argList = new LinkedList<String>();
    argList.add("-i");
    argList.add(inputPath.toString());
    argList.add("-o");
    argList.add(outputPath.toString());
    
    if (sequential) 
      argList.add("-seq");
    
    if (named)
      argList.add("-nv");
    
    String[] args = argList.toArray(new String[0]);
    
    SparseVectorsFromSequenceFiles.main(args);

    Path tfVectors = new Path(outputPath, "tf-vectors");
    Path tfidfVectors = new Path(outputPath, "tfidf-vectors");
    
    DictionaryVectorizerTest.validateVectors(fs, conf, NUM_DOCS, tfVectors, sequential, named);
    DictionaryVectorizerTest.validateVectors(fs, conf, NUM_DOCS, tfidfVectors, sequential, named);
  }  
}
