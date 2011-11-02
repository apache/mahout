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

package org.apache.mahout.df;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.google.common.io.Closeables;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.df.node.Node;
import org.apache.mahout.ga.watchmaker.OutputUtils;

/**
 * Utility class that contains various helper methods
 */
public final class DFUtils {
  private DFUtils() { }
  
  /**
   * Writes an Node[] into a DataOutput
   */
  public static void writeArray(DataOutput out, Node[] array) throws IOException {
    out.writeInt(array.length);
    for (Node w : array) {
      w.write(out);
    }
  }
  
  /**
   * Reads a Node[] from a DataInput
   */
  public static Node[] readNodeArray(DataInput in) throws IOException {
    int length = in.readInt();
    Node[] nodes = new Node[length];
    for (int index = 0; index < length; index++) {
      nodes[index] = Node.read(in);
    }
    
    return nodes;
  }
  
  /**
   * Writes a double[] into a DataOutput
   */
  public static void writeArray(DataOutput out, double[] array) throws IOException {
    out.writeInt(array.length);
    for (double value : array) {
      out.writeDouble(value);
    }
  }
  
  /**
   * Reads a double[] from a DataInput
   */
  public static double[] readDoubleArray(DataInput in) throws IOException {
    int length = in.readInt();
    double[] array = new double[length];
    for (int index = 0; index < length; index++) {
      array[index] = in.readDouble();
    }
    
    return array;
  }
  
  /**
   * Writes an int[] into a DataOutput
   */
  public static void writeArray(DataOutput out, int[] array) throws IOException {
    out.writeInt(array.length);
    for (int value : array) {
      out.writeInt(value);
    }
  }
  
  /**
   * Reads an int[] from a DataInput
   */
  public static int[] readIntArray(DataInput in) throws IOException {
    int length = in.readInt();
    int[] array = new int[length];
    for (int index = 0; index < length; index++) {
      array[index] = in.readInt();
    }
    
    return array;
  }
  
  /**
   * Return a list of all files in the output directory
   *
   * @throws IOException if no file is found
   */
  public static Path[] listOutputFiles(FileSystem fs, Path outputPath) throws IOException {
    Path[] outfiles = OutputUtils.listOutputFiles(fs, outputPath);
    if (outfiles.length == 0) {
      throw new IOException("No output found !");
    }
    
    return outfiles;
  }
  
  /**
   * Formats a time interval in milliseconds to a String in the form "hours:minutes:seconds:millis"
   */
  public static String elapsedTime(long milli) {
    long seconds = milli / 1000;
    milli %= 1000;
    
    long minutes = seconds / 60;
    seconds %= 60;
    
    long hours = minutes / 60;
    minutes %= 60;
    
    return hours + "h " + minutes + "m " + seconds + "s " + milli;
  }

  public static void storeWritable(Configuration conf, Path path, Writable writable) throws IOException {
    FileSystem fs = path.getFileSystem(conf);

    FSDataOutputStream out = fs.create(path);
    try {
      writable.write(out);
    } finally {
      Closeables.closeQuietly(out);
    }
  }
}
