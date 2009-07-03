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

package org.apache.mahout.utils.vectors;

import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.Vector;

import java.util.Iterator;
import java.io.IOException;


/**
 * Reads in a file containing {@link org.apache.mahout.matrix.Vector}s and provides
 * a {@link org.apache.mahout.utils.vectors.VectorIterable} interface to them.
 * <p/>
 * The key is any {@link org.apache.hadoop.io.Writable} and the value is a {@link org.apache.mahout.matrix.Vector}.
 * It can handle any class that implements Vector as long as it has a no-arg constructor.
 *
 **/
public class SequenceFileVectorIterable implements VectorIterable {
  private SequenceFile.Reader reader;
  private boolean transpose = false;

  public SequenceFileVectorIterable(SequenceFile.Reader reader) {
    this.reader = reader;
  }

  public SequenceFileVectorIterable(SequenceFile.Reader reader, boolean transpose) {
    this.reader = reader;
    this.transpose = transpose;
  }

  @Override
  public Iterator<Vector> iterator() {
    try {
      return new SeqFileIterator();
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
  }

  public class SeqFileIterator implements Iterator<Vector> {
    private Writable key;
    private Writable value;

    private SeqFileIterator() throws IllegalAccessException, InstantiationException {
      if (transpose == false){
        key = (Writable) reader.getKeyClass().newInstance();
        value = (Vector) reader.getValueClass().newInstance();
      } else {
        value = (Vector) reader.getValueClass().newInstance();
        key = (Writable) reader.getKeyClass().newInstance();
      }
    }

    @Override
    public boolean hasNext() {
      try {
        return reader.next(key, value);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public Vector next() {
      return transpose ? (Vector)key : (Vector)value;
    }

    /**
     * Only valid when {@link #next()} is also valid
     * @return The current Key
     */
    public Writable key(){
      return transpose ? value : key;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
