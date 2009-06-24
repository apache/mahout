package org.apache.mahout.clustering;

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;

import java.util.List;
import java.io.File;
import java.io.IOException;

/**
 *
 *
 **/
public class ClusteringTestUtils {

  public static void writePointsToFile(List<Vector> points, String fileName, FileSystem fs, Configuration conf)
          throws IOException {
    Path path = new Path(fileName);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, LongWritable.class, SparseVector.class);
    long recNum = 0;
    for (Vector point : points) {
      //point.write(dataOut);
      writer.append(new LongWritable(recNum++), point);
    }
    writer.close();
  }

  public static void rmr(String path) throws Exception {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (int i = 0; i < contents.length; i++)
          rmr(f.toString() + File.separator + contents[i]);
      }
      f.delete();
    }
  }
}
