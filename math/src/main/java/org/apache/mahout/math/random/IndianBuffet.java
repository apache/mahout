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

package org.apache.mahout.math.random;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.LineProcessor;
import com.google.common.io.Resources;
import org.apache.mahout.common.RandomUtils;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Samples a "document" from an IndianBuffet process.
 *
 * See http://mlg.eng.cam.ac.uk/zoubin/talks/turin09.pdf for details
 */
public final class IndianBuffet<T> implements Sampler<List<T>> {
  private final List<Integer> count = Lists.newArrayList();
  private int documents = 0;
  private final double alpha;
  private WordFunction<T> converter = null;
  private final Random gen;

  public IndianBuffet(double alpha, WordFunction<T> converter) {
    this.alpha = alpha;
    this.converter = converter;
    gen = RandomUtils.getRandom();
  }

  public static IndianBuffet<Integer> createIntegerDocumentSampler(double alpha) {
    return new IndianBuffet<Integer>(alpha, new IdentityConverter());
  }

  public static IndianBuffet<String> createTextDocumentSampler(double alpha) {
    return new IndianBuffet<String>(alpha, new WordConverter());
  }

  @Override
  public List<T> sample() {
    List<T> r = Lists.newArrayList();
    if (documents == 0) {
      double n = new PoissonSampler(alpha).sample();
      for (int i = 0; i < n; i++) {
        r.add(converter.convert(i));
        count.add(1);
      }
      documents++;
    } else {
      documents++;
      int i = 0;
      for (double cnt : count) {
        if (gen.nextDouble() < cnt / documents) {
          r.add(converter.convert(i));
          count.set(i, count.get(i) + 1);
        }
        i++;
      }
      int newItems = new PoissonSampler(alpha / documents).sample().intValue();
      for (int j = 0; j < newItems; j++) {
        r.add(converter.convert(i + j));
        count.add(1);
      }
    }
    return r;
  }

  private interface WordFunction<T> {
    T convert(int i);
  }

  /**
   * Just converts to an integer.
   */
  public static class IdentityConverter implements WordFunction<Integer> {
    @Override
    public Integer convert(int i) {
      return i;
    }
  }

  /**
   * Converts to a string.
   */
  public static class StringConverter implements WordFunction<String> {
    @Override
    public String convert(int i) {
      return String.valueOf(i);
    }
  }

  /**
   * Converts to one of a list of common English words for reasonably small integers and converts
   * to a token like w_92463 for big integers.
   */
  public static final class WordConverter implements WordFunction<String> {
    private final Splitter onSpace = Splitter.on(CharMatcher.WHITESPACE).omitEmptyStrings().trimResults();
    private final List<String> words;

    public WordConverter() {
      try {
        words = Resources.readLines(Resources.getResource("words.txt"), Charsets.UTF_8,
                                    new LineProcessor<List<String>>() {
            private final List<String> theWords = Lists.newArrayList();

            @Override
            public boolean processLine(String line) {
              Iterables.addAll(theWords, onSpace.split(line));
              return true;
            }

            @Override
            public List<String> getResult() {
              return theWords;
            }
          });
      } catch (IOException e) {
        throw new ImpossibleException(e);
      }
    }

    @Override
    public String convert(int i) {
      if (i < words.size()) {
        return words.get(i);
      } else {
        return "w_" + i;
      }
    }
  }

  public static class ImpossibleException extends RuntimeException {
    public ImpossibleException(Throwable e) {
      super(e);
    }
  }
}
