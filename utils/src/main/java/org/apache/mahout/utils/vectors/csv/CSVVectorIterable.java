package org.apache.mahout.utils.vectors.csv;
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

import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVStrategy;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.Reader;
import java.util.Iterator;
import java.util.NoSuchElementException;


/**
 * Iterates a CSV file and produces {@link org.apache.mahout.math.Vector}.
 * <br/>
 * The Iterator returned throws {@link UnsupportedOperationException} for the {@link java.util.Iterator#remove()} method.
 * <p/>
 * Assumes DenseVector for now, but in the future may have the option of mapping columns to sparse format
 * <p/>
 * The Iterator is not thread-safe.
 */
public class CSVVectorIterable implements Iterable<Vector> {

  private final CSVParser parser;
  private String [] line;

  public CSVVectorIterable(Reader reader) throws IOException {
    parser = new CSVParser(reader);
    line = parser.getLine();
  }

  public CSVVectorIterable(Reader reader, CSVStrategy strategy) throws IOException {
    parser = new CSVParser(reader, strategy);
    line = parser.getLine();
  }

  @Override
  public Iterator<Vector> iterator() {
    return new CSVIterator();
  }

  private class CSVIterator implements Iterator<Vector>{

    @Override
    public boolean hasNext() {
      return line != null;
    }

    @Override
    public Vector next() {

      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      Vector result = new DenseVector(line.length);
      for (int i = 0; i < line.length; i++) {
        result.setQuick(i, Double.parseDouble(line[i]));
      }
      //move the line forward
      try {
        line = parser.getLine();
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
