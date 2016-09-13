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

package org.apache.mahout.drivers

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.{ConfigMap, FunSuite}


class RowSimilarityDriverSuite extends FunSuite with DistributedSparkSuite  {

  val TextDocs = Array(
    "doc1\tNow is the time for all good people to come to aid of their party",
    "doc2\tNow is the time for all good people to come to aid of their country",
    "doc3\tNow is the time for all good people to come to aid of their hood",
    "doc4\tNow is the time for all good people to come to aid of their friends",
    "doc5\tNow is the time for all good people to come to aid of their looser brother",
    "doc6\tThe quick brown fox jumped over the lazy dog",
    "doc7\tThe quick brown fox jumped over the lazy boy",
    "doc8\tThe quick brown fox jumped over the lazy cat",
    "doc9\tThe quick brown fox jumped over the lazy wolverine",
    "doc10\tThe quick brown fox jumped over the lazy cantelope")// yes that's spelled correctly.

  test("RowSimilarityDriver text docs no strengths") {

    val firstFiveSimDocsTokens = tokenize(Iterable(
      "doc1\tdoc3 doc2 doc4 doc5"))

    val lastFiveSimDocsTokens = tokenize(Iterable(
      "doc6\tdoc8 doc10 doc7 doc9"))

    val inDir = TmpDir + "in-dir/"
    val inFilename = "in-file.tsv"
    val inPath = inDir + inFilename

    val outPath = TmpDir + "similarity-matrices/"


    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(TextDocs).coalesce(1, shuffle=true).saveAsTextFile(inDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir + "part-00000"), new Path(inPath))

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
      "--input", inPath,
      "--output", outPath,
      "--omitStrength",
      "--maxSimilaritiesPerRow", "4", // would get all docs similar if we didn't limit them
      "--master", masterUrl))

    val simLines = mahoutCtx.textFile(outPath).collect
    simLines.foreach { line =>
      val lineTokens = line.split("[\t ]")
      if (lineTokens.contains("doc1") ) // docs are two flavors so if only 4 similarities it will effectively classify
        lineTokens should contain theSameElementsAs firstFiveSimDocsTokens
      else
        lineTokens should contain theSameElementsAs lastFiveSimDocsTokens
    }

  }

  test("RowSimilarityDriver text docs") {

    val simDocsTokens = tokenize(Iterable(
      "doc1\tdoc3:27.87301122947484 doc2:27.87301122947484 doc4:27.87301122947484 doc5:23.42278065550721",
      "doc2\tdoc4:27.87301122947484 doc3:27.87301122947484 doc1:27.87301122947484 doc5:23.42278065550721",
      "doc3\tdoc4:27.87301122947484 doc2:27.87301122947484 doc1:27.87301122947484 doc5:23.42278065550721",
      "doc4\tdoc3:27.87301122947484 doc2:27.87301122947484 doc1:27.87301122947484 doc5:23.42278065550721",
      "doc5\tdoc4:23.42278065550721 doc2:23.42278065550721 doc3:23.42278065550721 doc1:23.42278065550721",
      "doc6\tdoc8:22.936393049704463 doc10:22.936393049704463 doc7:22.936393049704463 doc9:22.936393049704463",
      "doc7\tdoc6:22.936393049704463 doc8:22.936393049704463 doc10:22.936393049704463 doc9:22.936393049704463",
      "doc8\tdoc6:22.936393049704463 doc10:22.936393049704463 doc7:22.936393049704463 doc9:22.936393049704463",
      "doc9\tdoc6:22.936393049704463 doc8:22.936393049704463 doc10:22.936393049704463 doc7:22.936393049704463",
      "doc10\tdoc6:22.936393049704463 doc8:22.936393049704463 doc7:22.936393049704463 doc9:22.936393049704463"))

    val inDir = TmpDir + "in-dir/"
    val inFilename = "in-file.tsv"
    val inPath = inDir + inFilename

    val outPath = TmpDir + "similarity-matrix/"


    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(TextDocs).coalesce(1, shuffle=true).saveAsTextFile(inDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir + "part-00000"), new Path(inPath))

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
      "--input", inPath,
      "--output", outPath,
      "--maxSimilaritiesPerRow", "4", // would get all docs similar if we didn't limit them
      "--master", masterUrl))

    val simLines = mahoutCtx.textFile(outPath).collect
    tokenize(simLines) should contain theSameElementsAs simDocsTokens
  }

  // convert into an Iterable of tokens for 'should contain theSameElementsAs Iterable'
  def tokenize(a: Iterable[String], splitString: String = "[\t ]"): Iterable[String] = {
    var r: Iterable[String] = Iterable()
    a.foreach ( l => r = r ++ l.split(splitString) )
    r
  }

  override protected def beforeAll(configMap: ConfigMap) {
    super.beforeAll(configMap)
    RowSimilarityDriver.useContext(mahoutCtx)
  }

}
