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

/*
  final val matrixLLRCoocAtAControl = dense(
    (0.0,                0.6331745808516107, 0.0,                     0.0,                0.0),
    (0.6331745808516107, 0.0,                0.0,                     0.0,                0.0),
    (0.0,                0.0,                0.0,                     0.6331745808516107, 0.0),
    (0.0,                0.0,                0.6331745808516107,      0.0,                0.0),
    (0.0,                0.0,                0.0,                     0.0,                0.0))

  final val matrixLLRCoocBtAControl = dense(
      (1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 0.0),
      (0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.0),
      (0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.6795961471815897, 0.0),
      (1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 1.7260924347106847, 0.0),
      (0.0,                0.0,                0.6795961471815897, 0.0,                4.498681156950466))
*/


  final val SelfSimilairtyLines = Iterable(
      "galaxy\tnexus:1.7260924347106847",
      "ipad\tiphone:1.7260924347106847",
      "nexus\tgalaxy:1.7260924347106847",
      "iphone\tipad:1.7260924347106847",
      "surface")

  final val CrossIndicatorLines = Iterable(
      "iphone\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
      "ipad\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
      "nexus\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
      "galaxy\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
      "surface\tsurface:4.498681156950466 nexus:0.6795961471815897")

  // todo: a better test would be to sort each vector by itemID and compare rows, tokens misses some error cases
  final val SelfSimilairtyTokens = tokenize(SelfSimilairtyLines)

  final val SelfSimilairtyTokensOmitStrengths = tokenize(Iterable(
      "galaxy\tnexus",
      "ipad\tiphone",
      "nexus\tgalaxy",
      "iphone\tipad",
      "surface"))

  final val CrossIndicatorTokens = tokenize(CrossIndicatorLines)

  final val CrossIndicatorLinesWithoutStrengths = Array(
      "iphone\tnexus iphone ipad galaxy",
      "ipad\tnexus iphone ipad galaxy",
      "nexus\tnexus iphone ipad galaxy",
      "galaxy\tnexus iphone ipad galaxy",
      "surface\tsurface nexus")

  final val CrossIndicatorTokensOmitStrengths = tokenize(CrossIndicatorLinesWithoutStrengths)

  final val DrmALines = Array(
      "u1\tiphone:1 ipad:1",
      "u2\tnexus:1 galaxy:1",
      "u3\tsurface:1",
      "u4\tiphone:1 galaxy:1")

  /* interactions
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy"
  */

  final val DrmBLines = Array(
      "u1\tiphone:1 ipad:1 nexus:1 galaxy:1",
      "u2\tiphone:1 ipad:1 nexus:1 galaxy:1",
      "u3\tsurface:1 nexus:1",
      "u4\tiphone:1 ipad:1 galaxy:1")

  /* interactions
        "u1,view,iphone",
        "u1,view,ipad",
        "u1,view,nexus",
        "u1,view,galaxy",
        "u2,view,iphone",
        "u2,view,ipad",
        "u2,view,nexus",
        "u2,view,galaxy",
        "u3,view,surface",
        "u3,view,nexus",
        "u4,view,iphone",
        "u4,view,ipad",
        "u4,view,galaxy")
  */
  final val DrmALinesWitoutStrengths = Array(
      "u1\tiphone ipad",
      "u2\tnexus galaxy",
      "u3\tsurface",
      "u4\tiphone galaxy")

  final val DrmBLinesWitoutStrengths = Array(
      "u1\tiphone ipad nexus galaxy",
      "u2\tiphone ipad nexus galaxy",
      "u3\tsurface nexus",
      "u4\tiphone ipad galaxy")

  test("RowSimilarityDriver two matrices input") {

    val inDir = TmpDir + "in-dir/"
    val inFilename = "in-file.tsv"
    val inPath = inDir + inFilename
    val inDir2 = TmpDir + "in-dir2/"
    val inFilename2 = "in-file2.tsv"
    val inPath2 = inDir2 + inFilename2

    val outPath = TmpDir + "indicator-matrices"


    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(DrmALines).coalesce(1, shuffle=true).saveAsTextFile(inDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir + "part-00000"), new Path(inPath))

    mahoutCtx.parallelize(DrmBLines).coalesce(1, shuffle=true).saveAsTextFile(inDir2)

    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir2 + "part-00000"), new Path(inPath2))

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
      "--input", inPath,
      "--input2", inPath2,
      "--output", outPath,
      "--master", masterUrl))

    val indicatorLines = mahoutCtx.textFile(outPath + "/indicator-matrix/").collect.toIterable
    tokenize(indicatorLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossIndicatorLines = mahoutCtx.textFile(outPath + "/cross-indicator-matrix/").collect.toIterable
    tokenize(crossIndicatorLines) should contain theSameElementsAs CrossIndicatorTokens

  }

  test("RowSimilarityDriver custom delimiters") {

    val inFile = TmpDir + "in-file/"
    val inFile2 = TmpDir + "in-file2/"
    val outPath = TmpDir + "indicator-matrices/"

    val drmCustomDelimLinesA = Array(
      "u1-iphone;1=ipad;1",
      "u2-nexus;1=galaxy;1",
      "u3-surface;1",
      "u4-iphone;1=galaxy;1")

    val drmCustomDelimLinesB = Array(
      "u1-iphone;1=ipad;1=nexus;1=galaxy;1",
      "u2-iphone;1=ipad;1=nexus;1=galaxy;1",
      "u3-surface;1=nexus;1",
      "u4-iphone;1=ipad;1=galaxy;1")

    val customSelfIndicatorTokens = tokenize(Iterable(
      "galaxy-nexus;1.7260924347106847",
      "ipad-iphone;1.7260924347106847",
      "nexus-galaxy;1.7260924347106847",
      "iphone-ipad;1.7260924347106847",
      "surface"), "[-=]")

    val customCrossIndicatorTokens = tokenize(Iterable(
      "iphone-nexus;1.7260924347106847=iphone;1.7260924347106847=ipad;1.7260924347106847=galaxy;1.7260924347106847",
      "ipad-nexus;0.6795961471815897=iphone;0.6795961471815897=ipad;0.6795961471815897=galaxy;0.6795961471815897",
      "nexus-nexus;0.6795961471815897=iphone;0.6795961471815897=ipad;0.6795961471815897=galaxy;0.6795961471815897",
      "galaxy-nexus;1.7260924347106847=iphone;1.7260924347106847=ipad;1.7260924347106847=galaxy;1.7260924347106847",
      "surface-surface;4.498681156950466=nexus;0.6795961471815897"), "[-=]")

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesARdd = mahoutCtx.parallelize(drmCustomDelimLinesA).saveAsTextFile(inFile)
    val linesBRdd = mahoutCtx.parallelize(drmCustomDelimLinesB).saveAsTextFile(inFile2)

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
      "--input", inFile,
      "--input2", inFile2,
      "--output", outPath,
      "--master", masterUrl,
      "--rowKeyDelim", "-",
      "--columnIdStrengthDelim", ";",
      "--elementDelim", "="))

    // todo: a better test would be to get sorted vectors and compare rows instead of tokens, this might miss
    // some error cases
    val indicatorLines = mahoutCtx.textFile(outPath + "/indicator-matrix/").collect.toIterable
    tokenize(indicatorLines, "[-=]") should contain theSameElementsAs customSelfIndicatorTokens
    val crossIndicatorLines = mahoutCtx.textFile(outPath + "/cross-indicator-matrix/").collect.toIterable
    tokenize(crossIndicatorLines, "[-=]") should contain theSameElementsAs customCrossIndicatorTokens
  }

  test("RowSimilarityDriver write search engine output") {

    val inDir = TmpDir + "in-dir/"
    val inFilename = "in-file.tsv"
    val inPath = inDir + inFilename
    val inDir2 = TmpDir + "in-dir2/"
    val inFilename2 = "in-file2.tsv"
    val inPath2 = inDir2 + inFilename2

    val outPath = TmpDir + "indicator-matrices"


    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(DrmALinesWitoutStrengths).coalesce(1, shuffle=true).saveAsTextFile(inDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir + "part-00000"), new Path(inPath))

    mahoutCtx.parallelize(DrmBLinesWitoutStrengths).coalesce(1, shuffle=true).saveAsTextFile(inDir2)

    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir2 + "part-00000"), new Path(inPath2))

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
      "--input", inPath,
      "--input2", inPath2,
      "--output", outPath,
      "--master", masterUrl,
      "--omitStrength"))

    val indicatorLines = mahoutCtx.textFile(outPath + "/indicator-matrix/").collect.toIterable
    tokenize(indicatorLines) should contain theSameElementsAs SelfSimilairtyTokensOmitStrengths
    val crossIndicatorLines = mahoutCtx.textFile(outPath + "/cross-indicator-matrix/").collect.toIterable
    tokenize(crossIndicatorLines) should contain theSameElementsAs CrossIndicatorTokensOmitStrengths

  }

  test("RowSimilarityDriver recursive file discovery using filename patterns") {
    //directory structure using the following
    // tmp/data/a.tsv
    // tmp/data/more-data/another-dir/b.tsv

    val inFilename1 = "a.tsv"
    val inDir1 = TmpDir + "data/"
    val inPath1 = inDir1 + inFilename1
    val inFilename2 = "b.tsv"
    val inDir2 = TmpDir + "data/more-data/another-dir/"
    val inPath2 = inDir2 + inFilename2

    val inPathStart = TmpDir + "data/"
    val outPath = TmpDir + "indicator-matrices"

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(DrmALines).coalesce(1, shuffle=true).saveAsTextFile(inDir1)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(inDir1 + "part-00000"), new Path(inPath1))

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(DrmBLines).coalesce(1, shuffle=true).saveAsTextFile(inDir2)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    //rename part-00000 to tmp/some-location/something.tsv
    fs.rename(new Path(inDir2 + "part-00000"), new Path(inPath2))

    // local multi-threaded Spark with default FS, suitable for build tests but need better location for data

    RowSimilarityDriver.main(Array(
        "--input", inPathStart,
        "--input2", inPath2,
        "--output", outPath,
        "--filenamePattern", "a.tsv",
        "--master", masterUrl,
        "--recursive"))

    val indicatorLines = mahoutCtx.textFile(outPath + "/indicator-matrix/").collect.toIterable
    tokenize(indicatorLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossIndicatorLines = mahoutCtx.textFile(outPath + "/cross-indicator-matrix/").collect.toIterable
    tokenize(crossIndicatorLines) should contain theSameElementsAs CrossIndicatorTokens

  }

 test("RowSimilarityDriver cross similarity two separate items spaces, missing rows in B"){
    /* cross-similarity with category views, same user space
            	phones	tablets	mobile_acc	soap
            u1	0	      1	      1	          0
            u2	1	      1	      1	          0
removed ==> u3	0	      0	      1	          0
            u4	1	      1	      0	          1
    */
    val inFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val inFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val outPath = TmpDir + "indicator-matrices/"

    val drmBLinesSecondItemSet = Array(
        "u1\tphones:1 mobile_acc:1",
        "u2\tphones:1 tablets:1 mobile_acc:1",
        "u4\tphones:1 tablets:1 soap:1")
   
    /* interactions
        "u1,view,phones",
        "u1,view,mobile_acc",
        "u2,view,phones",
        "u2,view,tablets",
        "u2,view,mobile_acc",
        //"u3,view,mobile_acc",// if this line is removed the cross-cooccurrence should work
        "u4,view,phones",
        "u4,view,tablets",
        "u4,view,soap"
    */

    val UnequalDimensionsCrossSimilarityLines = tokenize(Iterable(
        "galaxy\ttablets:5.545177444479561 soap:1.7260924347106847 phones:1.7260924347106847",
        "ipad\tmobile_acc:1.7260924347106847 phones:0.6795961471815897",
        "surface",
        "nexus\tmobile_acc:1.7260924347106847 tablets:1.7260924347106847 phones:0.6795961471815897",
        "iphone\tsoap:1.7260924347106847 phones:1.7260924347106847"))

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(DrmALines).saveAsTextFile(inFile1)
    val linesRdd2 = mahoutCtx.parallelize(drmBLinesSecondItemSet).saveAsTextFile(inFile2)

    // local multi-threaded Spark with default HDFS
    RowSimilarityDriver.main(Array(
        "--input", inFile1,
        "--input2", inFile2,
        "--output", outPath,
        "--master", masterUrl,
        "--writeAllDatasets"))

    val indicatorLines = mahoutCtx.textFile(outPath + "/indicator-matrix/").collect.toIterable
    val crossIndicatorLines = mahoutCtx.textFile(outPath + "/cross-indicator-matrix/").collect.toIterable
    tokenize(indicatorLines) should contain theSameElementsAs SelfSimilairtyTokens
    tokenize(crossIndicatorLines) should contain theSameElementsAs UnequalDimensionsCrossSimilarityLines
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
