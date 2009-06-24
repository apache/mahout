package org.apache.mahout.utils.vectors;

import junit.framework.TestCase;

import java.io.File;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;


/**
 *
 *
 **/
public class SequenceFileVectorIterableTest extends TestCase {
  public void testIterable() throws Exception {
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    File tmpLoc = new File(tmpDir, "sfvit");
    tmpLoc.mkdirs();
    File tmpFile = File.createTempFile("sfvit", ".dat", tmpLoc);

    Path path = new Path(tmpFile.getAbsolutePath());
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, SparseVector.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    RandomVectorIterable iter = new RandomVectorIterable(50);
    writer.write(iter);
    writer.close();

    SequenceFile.Reader seqReader = new SequenceFile.Reader(fs, path, conf);
    SequenceFileVectorIterable sfvi = new SequenceFileVectorIterable(seqReader);
    int count = 0;
    for (Vector vector : sfvi) {
      System.out.println("Vec: " + vector.asFormatString());
      count++;
    }
    seqReader.close();
    assertTrue(count + " does not equal: " + 50, count == 50);
  }
}
