package org.apache.mahout.utils.vectors.io;
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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;


/**
 * Given a Sequence File containing vectors (actually, {@link org.apache.mahout.math.VectorWritable}, iterate over it.
 *
 **/
public class SequenceFileVectorIterable implements Iterable<Vector>{
  protected SequenceFile.Reader reader;
  protected long fileLen;
  protected Writable keyWritable;
  protected Writable valueWritable;
  protected boolean useKey;

  /**
   * Construct the Iterable
   * @param fs The {@link org.apache.hadoop.fs.FileSystem} containing the {@link org.apache.hadoop.io.SequenceFile}
   * @param file The {@link org.apache.hadoop.fs.Path} containing the file
   * @param conf The {@link org.apache.hadoop.conf.Configuration} to use
   * @param useKey If true, use the key as the {@link org.apache.mahout.math.VectorWritable}, otherwise use the value
   * @throws IllegalAccessException
   * @throws InstantiationException
   * @throws IOException
   */
  public SequenceFileVectorIterable(FileSystem fs, Path file, Configuration conf, boolean useKey) throws IllegalAccessException, InstantiationException, IOException {
    this.reader = new SequenceFile.Reader(fs, file, conf);
    ContentSummary summary = fs.getContentSummary(file);
    fileLen = summary.getLength();
    this.useKey = useKey;
    keyWritable = reader.getKeyClass().asSubclass(Writable.class).newInstance();
    valueWritable = reader.getValueClass().asSubclass(Writable.class).newInstance();
  }

  /**
   * The Iterator returned does not support remove()
   * @return The {@link java.util.Iterator}
   */
  public Iterator<Vector> iterator() {
    return new SFIterator();

  }

  private final class SFIterator implements Iterator<Vector>{
    @Override
    public boolean hasNext() {
      //TODO: is this legitimate?  We can't call next here since it breaks the iterator contract
      try {
        return reader.getPosition() < fileLen;
      } catch (IOException e) {
        return false;
      }
    }

    @Override
    public Vector next() {
      Vector result = null;
      boolean valid = false;
      try {
        valid = reader.next(keyWritable, valueWritable);
        if (valid){
          result = ((VectorWritable) (useKey ? keyWritable : valueWritable)).get();
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      return result;
    }

    /**
     * Not supported
     */
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
