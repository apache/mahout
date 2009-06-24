package org.apache.mahout.utils.vectors.io;

import junit.framework.TestCase;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.utils.vectors.RandomVectorIterable;

import java.io.File;


/**
 *
 *
 **/
public class SequenceFileVectorWriterTest extends TestCase {

  public void testSFVW() throws Exception {
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    File tmpLoc = new File(tmpDir, "sfvwt");
    tmpLoc.mkdirs();
    File tmpFile = File.createTempFile("sfvwt", ".dat", tmpLoc);

    Path path = new Path(tmpFile.getAbsolutePath());
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, SparseVector.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    RandomVectorIterable iter = new RandomVectorIterable(50);
    writer.write(iter);
    writer.close();

    SequenceFile.Reader seqReader = new SequenceFile.Reader(fs, path, conf);
    LongWritable key = new LongWritable();
    SparseVector value = new SparseVector();
    int count = 0;
    while (seqReader.next(key, value)){
      count++;
    }
    assertTrue(count + " does not equal: " + 50, count == 50);
  }
}
