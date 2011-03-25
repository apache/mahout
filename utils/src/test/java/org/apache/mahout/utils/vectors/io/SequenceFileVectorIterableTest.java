package org.apache.mahout.utils.vectors.io;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.MahoutTestCase;
import org.apache.mahout.utils.vectors.RandomVectorIterable;
import org.junit.Test;


/**
 *
 *
 **/
public class SequenceFileVectorIterableTest extends MahoutTestCase {


  @Test
  public void testSFVI() throws Exception {
    Path path = getTestTempFilePath("sfvw");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, VectorWritable.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    Iterable<Vector> iter = new RandomVectorIterable(50);
    writer.write(iter);
    writer.close();
    SequenceFileVectorIterable sfVIter = new SequenceFileVectorIterable(fs, path, conf, false);
    int count = 0;
    for (Vector vector : sfVIter) {
      count++;
    }
    assertEquals(50, count);
  }
}
