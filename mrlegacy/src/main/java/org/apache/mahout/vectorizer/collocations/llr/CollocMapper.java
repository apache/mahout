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

package org.apache.mahout.vectorizer.collocations.llr;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.lucene.IteratorTokenStream;
import org.apache.mahout.math.function.ObjectIntProcedure;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Pass 1 of the Collocation discovery job which generated ngrams and emits ngrams an their component n-1grams.
 * Input is a SequeceFile<Text,StringTuple>, where the key is a document id and the value is the tokenized documents.
 * <p/>
 */
public class CollocMapper extends Mapper<Text, StringTuple, GramKey, Gram> {

  private static final byte[] EMPTY = new byte[0];

  public static final String MAX_SHINGLE_SIZE = "maxShingleSize";

  private static final int DEFAULT_MAX_SHINGLE_SIZE = 2;

  public enum Count {
    NGRAM_TOTAL
  }

  private static final Logger log = LoggerFactory.getLogger(CollocMapper.class);

  private int maxShingleSize;

  private boolean emitUnigrams;

  /**
   * Collocation finder: pass 1 map phase.
   * <p/>
   * Receives a token stream which gets passed through a Lucene ShingleFilter. The ShingleFilter delivers ngrams of
   * the appropriate size which are then decomposed into head and tail subgrams which are collected in the
   * following manner
   * <p/>
   * <pre>
   * k:head_key,           v:head_subgram
   * k:head_key,ngram_key, v:ngram
   * k:tail_key,           v:tail_subgram
   * k:tail_key,ngram_key, v:ngram
   * </pre>
   * <p/>
   * The 'head' or 'tail' prefix is used to specify whether the subgram in question is the head or tail of the
   * ngram. In this implementation the head of the ngram is a (n-1)gram, and the tail is a (1)gram.
   * <p/>
   * For example, given 'click and clack' and an ngram length of 3:
   * <pre>
   * k: head_'click and'                         v:head_'click and'
   * k: head_'click and',ngram_'click and clack' v:ngram_'click and clack'
   * k: tail_'clack',                            v:tail_'clack'
   * k: tail_'clack',ngram_'click and clack'     v:ngram_'click and clack'
   * </pre>
   * <p/>
   * Also counts the total number of ngrams encountered and adds it to the counter
   * CollocDriver.Count.NGRAM_TOTAL
   * </p>
   *
   * @throws IOException if there's a problem with the ShingleFilter reading data or the collector collecting output.
   */
  @Override
  protected void map(Text key, StringTuple value, final Context context) throws IOException, InterruptedException {

    ShingleFilter sf = new ShingleFilter(new IteratorTokenStream(value.getEntries().iterator()), maxShingleSize);
    sf.reset();
    try {
      int count = 0; // ngram count

      OpenObjectIntHashMap<String> ngrams =
              new OpenObjectIntHashMap<String>(value.getEntries().size() * (maxShingleSize - 1));
      OpenObjectIntHashMap<String> unigrams = new OpenObjectIntHashMap<String>(value.getEntries().size());

      do {
        String term = sf.getAttribute(CharTermAttribute.class).toString();
        String type = sf.getAttribute(TypeAttribute.class).type();
        if ("shingle".equals(type)) {
          count++;
          ngrams.adjustOrPutValue(term, 1, 1);
        } else if (emitUnigrams && !term.isEmpty()) { // unigram
          unigrams.adjustOrPutValue(term, 1, 1);
        }
      } while (sf.incrementToken());

      final GramKey gramKey = new GramKey();

      ngrams.forEachPair(new ObjectIntProcedure<String>() {
        @Override
        public boolean apply(String term, int frequency) {
          // obtain components, the leading (n-1)gram and the trailing unigram.
          int i = term.lastIndexOf(' '); // TODO: fix for non-whitespace delimited languages.
          if (i != -1) { // bigram, trigram etc

            try {
              Gram ngram = new Gram(term, frequency, Gram.Type.NGRAM);
              Gram head = new Gram(term.substring(0, i), frequency, Gram.Type.HEAD);
              Gram tail = new Gram(term.substring(i + 1), frequency, Gram.Type.TAIL);

              gramKey.set(head, EMPTY);
              context.write(gramKey, head);

              gramKey.set(head, ngram.getBytes());
              context.write(gramKey, ngram);

              gramKey.set(tail, EMPTY);
              context.write(gramKey, tail);

              gramKey.set(tail, ngram.getBytes());
              context.write(gramKey, ngram);

            } catch (IOException e) {
              throw new IllegalStateException(e);
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
          }
          return true;
        }
      });

      unigrams.forEachPair(new ObjectIntProcedure<String>() {
        @Override
        public boolean apply(String term, int frequency) {
          try {
            Gram unigram = new Gram(term, frequency, Gram.Type.UNIGRAM);
            gramKey.set(unigram, EMPTY);
            context.write(gramKey, unigram);
          } catch (IOException e) {
            throw new IllegalStateException(e);
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
          return true;
        }
      });

      context.getCounter(Count.NGRAM_TOTAL).increment(count);
      sf.end();
    } finally {
      Closeables.close(sf, true);
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    this.maxShingleSize = conf.getInt(MAX_SHINGLE_SIZE, DEFAULT_MAX_SHINGLE_SIZE);

    this.emitUnigrams = conf.getBoolean(CollocDriver.EMIT_UNIGRAMS, CollocDriver.DEFAULT_EMIT_UNIGRAMS);

    if (log.isInfoEnabled()) {
      log.info("Max Ngram size is {}", this.maxShingleSize);
      log.info("Emit Unitgrams is {}", emitUnigrams);
    }
  }

}
