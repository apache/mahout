/*
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

package org.apache.mahout.vectorizer.encoders;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.lucene.TokenStreamIterator;

import java.io.IOException;
import java.io.Reader;
import java.nio.CharBuffer;
import java.util.Iterator;

/**
 * Encodes text using a lucene style tokenizer.
 *
 * @see TextValueEncoder
 */
public class LuceneTextValueEncoder extends TextValueEncoder {
  private Analyzer analyzer;

  public LuceneTextValueEncoder(String name) {
    super(name);
  }

  public void setAnalyzer(Analyzer analyzer) {
    this.analyzer = analyzer;
  }

  /**
   * Tokenizes a string using the simplest method.  This should be over-ridden for more subtle
   * tokenization.
   */
  @Override
  protected Iterable<String> tokenize(CharSequence originalForm) {
    try {
      TokenStream ts = analyzer.tokenStream(getName(), new CharSequenceReader(originalForm));
      ts.addAttribute(CharTermAttribute.class);
      return new LuceneTokenIterable(ts, false);
    } catch (IOException ex) {
      throw new IllegalStateException(ex);
    }
  }

  private static final class CharSequenceReader extends Reader {
    private final CharBuffer buf;

    /**
     * Creates a new character-stream reader whose critical sections will synchronize on the reader
     * itself.
     */
    private CharSequenceReader(CharSequence input) {
      int n = input.length();
      buf = CharBuffer.allocate(n);
      for (int i = 0; i < n; i++) {
        buf.put(input.charAt(i));
      }
      buf.rewind();
    }

    /**
     * Reads characters into a portion of an array.  This method will block until some input is
     * available, an I/O error occurs, or the end of the stream is reached.
     *
     * @param cbuf Destination buffer
     * @param off  Offset at which to start storing characters
     * @param len  Maximum number of characters to read
     * @return The number of characters read, or -1 if the end of the stream has been reached
     */
    @Override
    public int read(char[] cbuf, int off, int len) {
      int toRead = Math.min(len, buf.remaining());
      if (toRead > 0) {
        buf.get(cbuf, off, toRead);
        return toRead;
      } else {
        return -1;
      }
    }

    @Override
    public void close() {
      // do nothing
    }
  }

  private static final class LuceneTokenIterable implements Iterable<String> {
    private boolean firstTime = true;
    private final TokenStream tokenStream;

    private LuceneTokenIterable(TokenStream ts, boolean firstTime) {
      this.tokenStream = ts;
      this.firstTime = firstTime;
    }

    /**
     * Returns an iterator over a set of elements of type T.
     *
     * @return an Iterator.
     */
    @Override
    public Iterator<String> iterator() {
      if (firstTime) {
        firstTime = false;
      } else {
        try {
          tokenStream.reset();
        } catch (IOException e) {
          throw new IllegalStateException("This token stream can't be reset");
        }
      }

      return new TokenStreamIterator(tokenStream);
    }
  }

}
