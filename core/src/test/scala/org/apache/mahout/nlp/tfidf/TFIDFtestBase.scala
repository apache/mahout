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

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}
import scala.collection._
import RLikeOps._
import scala.math._


trait TFIDFtestBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  val epsilon = 1E-6

  val documents: List[(Int, String)] = List(
    (1, "the first document contains 5 terms"),
    (2, "document two document contains 4 terms"),
    (3, "document three three terms"),
    (4, "each document including this document contain the term document"))

  def createDictionaryAndDfMaps(documents: List[(Int, String)]): (Map[String, Int], Map[Int, Int]) = {

    // get a tf count for the entire dictionary
    val dictMap = documents.unzip._2.mkString(" ").toLowerCase.split(" ").groupBy(identity).mapValues(_.length)

    // create a dictionary with an index for each term
    val dictIndex = dictMap.zipWithIndex.map(x => x._1._1 -> x._2)

    val docFrequencyCount = new Array[Int](dictMap.size)

    for (token <- dictMap) {
      for (doc <- documents) {
        // parse the string and get a word then increment the df count for that word
        if (doc._2.toLowerCase.split(" ").contains(token._1)) {
          docFrequencyCount(dictIndex(token._1)) += 1
        }
      }
    }

    val docFrequencyMap = docFrequencyCount.zipWithIndex.map(x => x._2 -> x._1).toMap

    (dictIndex, docFrequencyMap)
  }

  def vectorizeDocument(document: String,
                        dictionaryMap: Map[String, Int],
                        dfMap: Map[Int, Int], weight: TermWeight = new TFIDF): Vector = {

    val wordCounts = document.toLowerCase.split(" ").groupBy(identity).mapValues(_.length)

    val vec = new RandomAccessSparseVector(dictionaryMap.size)

    val totalDFSize = dictionaryMap.size
    val docSize = wordCounts.size

    for (word <- wordCounts) {
      val term = word._1
      if (dictionaryMap.contains(term)) {
        val termFreq = word._2
        val dictIndex = dictionaryMap(term)
        val docFreq = dfMap(dictIndex)
        val currentWeight = weight.calculate(termFreq, docFreq.toInt, docSize, totalDFSize.toInt)
        vec(dictIndex)= currentWeight
      }
    }
    vec
  }

  test("TF test") {

    val (dictionary, dfMap) = createDictionaryAndDfMaps(documents)

    val tf: TermWeight = new TF()

    val vectorizedDocuments: Matrix = new SparseMatrix(documents.size, dictionary.size)

    for (doc <- documents) {
      vectorizedDocuments(doc._1 - 1, ::) := vectorizeDocument(doc._2, dictionary, dfMap, tf)
    }

    // corpus:
    //  (1, "the first document contains 5 terms"),
    //  (2, "document two document contains 4 terms"),
    //  (3, "document three three terms"),
    //  (4, "each document including this document contain the term document")

    // dictonary:
    //  (this -> 0, 4 -> 1, three -> 2, document -> 3, two -> 4, term -> 5, 5 -> 6, contain -> 7,
    //   each -> 8, first -> 9, terms -> 10, contains -> 11, including -> 12, the -> 13)

    // dfMap:
    //  (0 -> 1, 5 -> 1, 10 -> 3, 1 -> 1, 6 -> 1, 9 -> 1, 13 -> 2, 2 -> 1, 12 -> 1, 7 -> 1, 3 -> 4,
    //   11 -> 2, 8 -> 1, 4 -> 1)

    vectorizedDocuments(0, 0).toInt should be (0)
    vectorizedDocuments(0, 13).toInt should be (1)
    vectorizedDocuments(1, 3).toInt should be (2)
    vectorizedDocuments(3, 3).toInt should be (3)

  }


  test("TFIDF test") {
    val (dictionary, dfMap) = createDictionaryAndDfMaps(documents)

    val tfidf: TermWeight = new TFIDF()

    val vectorizedDocuments: Matrix = new SparseMatrix(documents.size, dictionary.size)

    for (doc <- documents) {
      vectorizedDocuments(doc._1 - 1, ::) := vectorizeDocument(doc._2, dictionary, dfMap, tfidf)
    }

    // corpus:
    //  (1, "the first document contains 5 terms"),
    //  (2, "document two document contains 4 terms"),
    //  (3, "document three three terms"),
    //  (4, "each document including this document contain the term document")

    // dictonary:
    //  (this -> 0, 4 -> 1, three -> 2, document -> 3, two -> 4, term -> 5, 5 -> 6, contain -> 7,
    //   each -> 8, first -> 9, terms -> 10, contains -> 11, including -> 12, the -> 13)

    // dfMap:
    //  (0 -> 1, 5 -> 1, 10 -> 3, 1 -> 1, 6 -> 1, 9 -> 1, 13 -> 2, 2 -> 1, 12 -> 1, 7 -> 1, 3 -> 4,
    //   11 -> 2, 8 -> 1, 4 -> 1)

    abs(vectorizedDocuments(0, 0) -  0.0) should be < epsilon
    abs(vectorizedDocuments(0, 13) - 2.540445) should be < epsilon
    abs(vectorizedDocuments(1, 3) - 2.870315) should be < epsilon
    abs(vectorizedDocuments(3, 3) - 3.515403) should be < epsilon
  }

  test("MLlib TFIDF test") {
    val (dictionary, dfMap) = createDictionaryAndDfMaps(documents)

    val tfidf: TermWeight = new MLlibTFIDF()

    val vectorizedDocuments: Matrix = new SparseMatrix(documents.size, dictionary.size)

    for (doc <- documents) {
      vectorizedDocuments(doc._1 - 1, ::) := vectorizeDocument(doc._2, dictionary, dfMap, tfidf)
    }

    // corpus:
    //  (1, "the first document contains 5 terms"),
    //  (2, "document two document contains 4 terms"),
    //  (3, "document three three terms"),
    //  (4, "each document including this document contain the term document")

    // dictonary:
    //  (this -> 0, 4 -> 1, three -> 2, document -> 3, two -> 4, term -> 5, 5 -> 6, contain -> 7,
    //   each -> 8, first -> 9, terms -> 10, contains -> 11, including -> 12, the -> 13)

    // dfMap:
    //  (0 -> 1, 5 -> 1, 10 -> 3, 1 -> 1, 6 -> 1, 9 -> 1, 13 -> 2, 2 -> 1, 12 -> 1, 7 -> 1, 3 -> 4,
    //   11 -> 2, 8 -> 1, 4 -> 1)

    abs(vectorizedDocuments(0, 0) -  0.0) should be < epsilon
    abs(vectorizedDocuments(0, 13) - 1.609437) should be < epsilon
    abs(vectorizedDocuments(1, 3) - 2.197224) should be < epsilon
    abs(vectorizedDocuments(3, 3) - 3.295836) should be < epsilon
  }

}