/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.h2obindings.drm.H2ODrm;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import water.Futures;
import water.fvec.Frame;
import water.fvec.Vec;
import water.parser.ValueString;
import water.util.FrameUtils;

import java.io.File;
import java.io.IOException;
import java.net.URI;

/**
 * SequenceFile I/O class (on HDFS)
 */
public class H2OHdfs {
  /**
   * Predicate to check if a given filename is a SequenceFile.
   *
   * Inspect the first three bytes to determine the format of the file.
   *
   * @param filename Name of the file to check.
   * @return True if file is of SequenceFile format.
   */
  public static boolean isSeqfile(String filename) {
    try {
      Configuration conf = new Configuration();
      Path path = new Path(filename);
      FileSystem fs = FileSystem.get(URI.create(filename), conf);
      FSDataInputStream fin = fs.open(path);
      byte seq[] = new byte[3];

      fin.read(seq);
      fin.close();

      return seq[0] == 'S' && seq[1] == 'E' && seq[2] == 'Q';
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * Create DRM from SequenceFile.
   *
   * Create a Mahout DRM backed on H2O from the specified SequenceFile.
   *
   * @param filename Name of the sequence file.
   * @param parMin Minimum number of data partitions in the DRM.
   * @return DRM object created.
   */
  public static H2ODrm drmFromFile(String filename, int parMin) {
    try {
      if (isSeqfile(filename)) {
        return drmFromSeqfile(filename, parMin);
      } else {
        return new H2ODrm(FrameUtils.parseFrame(null,new File(filename)));
      }
    } catch (IOException e) {
      return null;
    }
  }

  /**
   * Internal method called from <code>drmFromFile</code> if format verified.
   */
  public static H2ODrm drmFromSeqfile(String filename, int parMin) {
    long rows = 0;
    int cols = 0;
    Frame frame = null;
    Vec labels = null;

    SequenceFile.Reader reader = null;
    try {
      Configuration conf = new Configuration();
      Path path = new Path(filename);
      FileSystem fs = FileSystem.get(URI.create(filename), conf);
      Vec.Writer writers[];
      Vec.Writer labelwriter = null;
      boolean isIntKey = false, isLongKey = false, isStringKey = false;

      reader = new SequenceFile.Reader(fs, path, conf);

      if (reader.getValueClass() != VectorWritable.class) {
        System.out.println("ValueClass in file " + filename +
                           "must be VectorWritable, but found " +
                           reader.getValueClassName());
        return null;
      }

      Writable key = (Writable)
        ReflectionUtils.newInstance(reader.getKeyClass(), conf);
      VectorWritable value = (VectorWritable)
        ReflectionUtils.newInstance(reader.getValueClass(), conf);

      long start = reader.getPosition();

      if (reader.getKeyClass() == Text.class) {
        isStringKey = true;
      } else if (reader.getKeyClass() == LongWritable.class) {
        isLongKey = true;
      } else {
        isIntKey = true;
      }

      while (reader.next(key, value)) {
        if (cols == 0) {
          Vector v = value.get();
          cols = Math.max(v.size(), cols);
        }
        if (isLongKey) {
          rows = Math.max(((LongWritable)(key)).get()+1, rows);
        }
        if (isIntKey) {
          rows = Math.max(((IntWritable)(key)).get()+1, rows);
        }
        if (isStringKey) {
          rows++;
        }
      }
      reader.seek(start);

      frame = H2OHelper.emptyFrame(rows, cols, parMin, -1);
      writers = new Vec.Writer[cols];
      for (int i = 0; i < writers.length; i++) {
        writers[i] = frame.vecs()[i].open();
      }

      if (reader.getKeyClass() == Text.class) {
        labels = H2OHelper.makeEmptyStrVec(frame.anyVec());
        labelwriter = labels.open();
      }

      long r = 0;
      while (reader.next(key, value)) {
        Vector v = value.get();
        if (isLongKey) {
          r = ((LongWritable)(key)).get();
        }
        if (isIntKey) {
          r = ((IntWritable)(key)).get();
        }
        for (int c = 0; c < v.size(); c++) {
          writers[c].set(r, v.getQuick(c));
        }
        if (labels != null) {
          labelwriter.set(r, (key).toString());
        }
        if (isStringKey) {
          r++;
        }
      }

      Futures fus = new Futures();
      for (Vec.Writer w : writers) {
        w.close(fus);
      }
      if (labelwriter != null) {
        labelwriter.close(fus);
      }
      fus.blockForPending();
    } catch (java.io.IOException e) {
      return null;
    } finally {
      IOUtils.closeStream(reader);
    }
    return new H2ODrm(frame, labels);
  }

  /**
   * Create SequenceFile on HDFS from DRM object.
   *
   * @param filename Filename to create and store DRM data in.
   * @param drm DRM object storing Matrix data in memory.
   */
  public static void drmToFile(String filename, H2ODrm drm) throws java.io.IOException {
    Frame frame = drm.frame;
    Vec labels = drm.keys;
    Configuration conf = new Configuration();
    Path path = new Path(filename);
    FileSystem fs = FileSystem.get(URI.create(filename), conf);
    SequenceFile.Writer writer;
    boolean isSparse = H2OHelper.isSparse(frame);
    ValueString vstr = new ValueString();

    if (labels != null) {
      writer = SequenceFile.createWriter(fs, conf, path, Text.class, VectorWritable.class);
    } else {
      writer = SequenceFile.createWriter(fs, conf, path, IntWritable.class, VectorWritable.class);
    }

    for (long r = 0; r < frame.anyVec().length(); r++) {
      Vector v;
      if (isSparse) {
        v = new SequentialAccessSparseVector(frame.numCols());
      } else {
        v = new DenseVector(frame.numCols());
      }

      for (int c = 0; c < frame.numCols(); c++) {
        v.setQuick(c, frame.vecs()[c].at(r));
      }

      if (labels != null) {
        writer.append(new Text(labels.atStr(vstr, r).toString()), new VectorWritable(v));
      } else {
        writer.append(new IntWritable((int)r), new VectorWritable(v));
      }
    }

    writer.close();
  }
}
