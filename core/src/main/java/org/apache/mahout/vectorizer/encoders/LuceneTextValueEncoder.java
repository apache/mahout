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
import org.apache.lucene.analysis.tokenattributes.TermAttribute;

import java.io.IOException;
import java.io.Reader;
import java.nio.CharBuffer;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Encodes text using a lucene style tokenizer.
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
    TokenStream ts = analyzer.tokenStream(getName(), new CharSequenceReader(originalForm));
    ts.addAttribute(TermAttribute.class);
    return new LuceneTokenIterable(ts);
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
      buf.get(cbuf, off, len);
      return len;
    }

    @Override
    public void close()  {
      // do nothing
    }
  }

  private static final class LuceneTokenIterable implements Iterable<String> {
    private boolean firstTime = true;
    private final TokenStream tokenStream;

    private LuceneTokenIterable(TokenStream ts) {
      this.tokenStream = ts;
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

  private static final class TokenStreamIterator implements Iterator<String> {
    private final TokenStream tokenStream;
    private String bufferedToken;

    private TokenStreamIterator(TokenStream tokenStream) {
      this.tokenStream = tokenStream;
    }

    /**
     * Returns <tt>true</tt> if the iteration has more elements. (In other words, returns <tt>true</tt>
     * if <tt>next</tt> would return an element rather than throwing an exception.)
     *
     * @return <tt>true</tt> if the iterator has more elements.
     */
    @Override
    public boolean hasNext() {
      if (bufferedToken == null) {
        boolean r;
        try {
          r = tokenStream.incrementToken();
        } catch (IOException e) {
          throw new TokenizationException("IO error while tokenizing", e);
        }
        if (r) {
          bufferedToken = tokenStream.getAttribute(TermAttribute.class).term();
        }
        return r;
      } else {
        return true;
      }
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration.
     * @throws NoSuchElementException iteration has no more elements.
     */
    @Override
    public String next() {
      if (bufferedToken != null) {
        String r = bufferedToken;
        bufferedToken = null;
        return r;
      } else if (hasNext()) {
        return next();
      } else {
        throw new NoSuchElementException("Ran off end if token stream");
      }
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException("Can't remove tokens");
    }
  }

  private static final class TokenizationException extends RuntimeException {
    private TokenizationException(String msg, Throwable cause) {
      super(msg, cause);
    }
  }
}
