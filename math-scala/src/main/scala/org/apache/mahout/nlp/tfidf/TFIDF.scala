package org.apache.mahout.nlp.tfidf

trait Weight {

  /**
   * @param tf term freq
   * @param df doc freq
   * @param length Length of the document
   * @param numDocs the total number of docs
   * @return The weight
   */
  def calculate(tf: Int, df: Int, length: Int, numDocs: Int): Double
}


class TFIDF extends Weight {

  /**
   * Calculate TF-IDF weight
   *
   * Lucene DefaultSimilarity's TF-IDF calculation uses the formula:
   *
   *   sqrt(termFreq) * log((numDocs / (docFreq + 1)) + 1.0)
   *
   * Note: this is consistent with the MapReduce seq2sparse implementation of TF-IDF weights
   * and is slightly different from Spark's TD-IDF calculation which is implemented as:
   *
   *   termFreq * log((numDocs + 1.0) / (docFreq + 1.0))
   *
   *
   * @param tf term freq
   * @param df doc freq
   * @param length Length of the document - UNUSED
   * @param numDocs the total number of docs
   * @return The TF-IDF weight as calculated by DefaultSimilarity
   */
  def calculate(tf: Int, df: Int, length: Int, numDocs: Int): Double = {

    // Lucene DefaultSimilarity's TF-IDF is implemented as:
    // sqrt(tf) * (log(numDocs/(df+1)) + 1)
    math.sqrt(tf) * (math.log(numDocs / (df + 1).toDouble) + 1.0)
  }
}

class TF extends Weight {

  /**
   * For TF Weight simply return the absolute TF.  We do not use the DefaultSimilarity's TF-IDF calculation
   * which returns:
   *
   *  sqrt(tf)
   *
   * this is consistent with the MapReduce seq2sparse implementation of TF-IDF weights.
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


