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

import java.io.IOException;
import java.net.URI;

import scala.Tuple2;

import water.fvec.Frame;
import water.fvec.Vec;
import water.Futures;
import water.parser.ValueString;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;



public class H2OHdfs {
  public static Tuple2<Frame,Vec> drm_from_file(String filename, int parMin) {
    long rows = 0;
    int cols = 0;
    Frame frame = null;
    Vec labels = null;

    SequenceFile.Reader reader = null;
    try {
      String uri = filename;
      Configuration conf = new Configuration();
      Path path = new Path(uri);
      FileSystem fs = FileSystem.get(URI.create(uri), conf);
      Vec.Writer writers[];
      Vec.Writer labelwriter = null;

      reader = new SequenceFile.Reader(fs, path, conf);

      if (reader.getValueClass() != VectorWritable.class) {
        System.out.println("ValueClass in file " + filename + "must be VectorWritable, but found " + reader.getValueClassName());
        return null;
      }

      Writable key = (Writable)
        ReflectionUtils.newInstance(reader.getKeyClass(), conf);
      VectorWritable value = (VectorWritable)
        ReflectionUtils.newInstance(reader.getValueClass(), conf);

      long start = reader.getPosition();
      while (reader.next(key, value)) {
        if (cols == 0) {
          Vector v = value.get();
          cols = v.size();
        }
        rows++;
      }
      reader.seek(start);

      frame = H2OHelper.empty_frame(rows, cols, parMin, -1);
      writers = new Vec.Writer[cols];
      for (int i = 0; i < writers.length; i++)
        writers[i] = frame.vecs()[i].open();

      if (reader.getKeyClass() == Text.class) {
        labels = frame.anyVec().makeZero();
        labelwriter = labels.open();
      }

      long r = 0;
      while (reader.next(key, value)) {
        Vector v = value.get();
        for (int c = 0; c < v.size(); c++)
          writers[c].set(r, v.getQuick(c));
        if (labels != null)
          labelwriter.set(r, ((Text)key).toString());
        r++;
      }

      Futures fus = new Futures();
      for (Vec.Writer w : writers)
        w.close(fus);
      if (labelwriter != null)
        labelwriter.close(fus);
      fus.blockForPending();
    } catch (java.io.IOException e) {
      return null;
    } finally {
      IOUtils.closeStream(reader);
    }
    return new Tuple2<Frame,Vec>(frame, labels);
  }

  public static void drm_to_file(String filename, Frame frame, Vec labels) throws java.io.IOException {
    String uri = filename;
    Configuration conf = new Configuration();
    Path path = new Path(uri);
    FileSystem fs = FileSystem.get(URI.create(uri), conf);
    SequenceFile.Writer writer = null;
    boolean is_sparse = H2OHelper.is_sparse(frame);
    ValueString vstr = new ValueString();

    if (labels != null)
      writer = SequenceFile.createWriter(fs, conf, path, Text.class, VectorWritable.class);
    else
      writer = SequenceFile.createWriter(fs, conf, path, IntWritable.class, VectorWritable.class);

    for (long r = 0; r < frame.anyVec().length(); r++) {
      Vector v = null;
      if (is_sparse)
        v = new SequentialAccessSparseVector(frame.numCols());
      else
        v = new DenseVector(frame.numCols());

      for (int c = 0; c < frame.numCols(); c++)
        v.setQuick(c, frame.vecs()[c].at(r));

      if (labels != null)
        writer.append(new Text(labels.atStr(vstr, r).toString()), new VectorWritable(v));
      else
        writer.append(new IntWritable((int)r), new VectorWritable(v));
    }

    writer.close();
  }
}
