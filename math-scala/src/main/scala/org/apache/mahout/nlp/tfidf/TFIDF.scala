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

package org.apache.mahout.nlp.tfidf

trait TermWeight {

  /**
   * @param tf term freq
   * @param df doc freq
   * @param length Length of the document
   * @param numDocs the total number of docs
   */
  def calculate(tf: Int, df: Int, length: Int, numDocs: Int): Double
}


class TFIDF extends TermWeight {

  /**
   * Calculate TF-IDF weight.
   *
   * Lucene 4.6's DefaultSimilarity TF-IDF calculation uses the formula:
   *
   *   sqrt(termFreq) * (log(numDocs / (docFreq + 1)) + 1.0)
   *
   * Note: this is consistent with the MapReduce seq2sparse implementation of TF-IDF weights
   * and is slightly different from Spark MLlib's TD-IDF calculation which is implemented as:
   *
   *   termFreq * log((numDocs + 1.0) / (docFreq + 1.0))
   *
   * @param tf term freq
   * @param df doc freq
   * @param length Length of the document - UNUSED
   * @param numDocs the total number of docs
   * @return The TF-IDF weight as calculated by Lucene 4.6's DefaultSimilarity
   */
  def calculate(tf: Int, df: Int, length: Int, numDocs: Int): Double = {

    // Lucene 4.6 DefaultSimilarity's TF-IDF is implemented as:
    // sqrt(tf) * (log(numDocs / (df + 1)) + 1)
    math.sqrt(tf) * (math.log(numDocs / (df + 1).toDouble) + 1.0)
  }
}

class MLlibTFIDF extends TermWeight {

  /**
   * Calculate TF-IDF weight with IDF formula used by Spark MLlib's IDF:
   *
   *   termFreq * log((numDocs + 1.0) / (docFreq + 1.0))
   *
   * Use this weight if working with MLLib vectorized documents.
   *
   * Note: this is not consistent with the MapReduce seq2sparse implementation of TF-IDF weights
   * which is implemented using Lucene DefaultSimilarity's TF-IDF calculation:
   *
   *   sqrt(termFreq) * (log(numDocs / (docFreq + 1)) + 1.0)
   *
   * @param tf term freq
   * @param df doc freq
   * @param length Length of the document - UNUSED
   * @param numDocs the total number of docs
   * @return The TF-IDF weight as calculated by Spark MLlib's IDF
   */
  def calculate(tf: Int, df: Int, length: Int, numDocs: Int): Double = {

    // Spark MLLib's TF-IDF weight is implemented as:
    // termFreq * log((numDocs + 1.0) / (docFreq + 1.0))
    tf *  math.log((numDocs + 1.0) / (df + 1).toDouble)
  }
}

class TF extends TermWeight {

  /**
   * For TF Weight simply return the absolute TF.
   *
   * Note: We do not use Lucene 4.6's DefaultSimilarity's TF calculation here
   * which returns:
   *
   *   sqrt(tf)
   *
   * this is consistent with the MapReduce seq2sparse implementation of TF weights.
   *
   * @param tf term freq
   * @param df doc freq - UNUSED
   * @param length Length of the document - UNUSED
   * @param numDocs the total number of docs - UNUSED
   * based on term frequency only - UNUSED
   * @return The weight = tf param
   */
  def calculate(tf: Int, df: Int = -Int.MaxValue, length: Int = -Int.MaxValue, numDocs: Int = -Int.MaxValue): Double = {
    tf
  }
}


