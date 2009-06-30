package org.apache.mahout.utils;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.kmeans.KMeansDriver;

import java.io.IOException;


/**
 *
 *
 **/
public class HadoopUtil {
  private transient static Log log = LogFactory.getLog(HadoopUtil.class);

  public static void overwriteOutput(String output) throws IOException {
    JobConf conf = new JobConf(KMeansDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    boolean exists = fs.exists(outPath);
    if (exists == true) {
      log.warn("Deleting " + outPath);
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);

  }
}
