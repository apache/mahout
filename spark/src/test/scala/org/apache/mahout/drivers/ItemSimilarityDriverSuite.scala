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
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.mahout.math.indexeddataset.{BiDictionary, IndexedDataset}
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark
import org.scalatest.{ConfigMap, FunSuite}
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._

import scala.collection.immutable.HashMap

//todo: take out, only for temp tests

import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import scala.collection.JavaConversions._


class ItemSimilarityDriverSuite extends FunSuite with DistributedSparkSuite {

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

  val CrossSimilarityLines = Iterable(
    "iphone\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
    "ipad\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
    "nexus\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
    "galaxy\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
    "surface\tsurface:4.498681156950466 nexus:0.6795961471815897")

  // todo: a better test would be to sort each vector by itemID and compare rows, tokens misses some error cases
  final val SelfSimilairtyTokens = tokenize(Iterable(
    "galaxy\tnexus:1.7260924347106847",
    "ipad\tiphone:1.7260924347106847",
    "nexus\tgalaxy:1.7260924347106847",
    "iphone\tipad:1.7260924347106847",
    "surface"))

  val CrossSimilarityTokens = tokenize(Iterable(
    "iphone\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
    "ipad\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
    "nexus\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897",
    "galaxy\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847",
    "surface\tsurface:4.498681156950466 nexus:0.6795961471815897"))

  /*
    //Clustered Spark and HDFS, not a good everyday build test
    ItemSimilarityDriver.main(Array(
        "--input", "hdfs://occam4:54310/user/pat/spark-itemsimilarity/cf-data.txt",
        "--output", "hdfs://occam4:54310/user/pat/spark-itemsimilarity/similarityMatrices/",
        "--master", "spark://occam4:7077",
        "--filter1", "purchase",
        "--filter2", "view",
        "--inDelim", ",",
        "--itemIDColumn", "2",
        "--rowIDColumn", "0",
        "--filterColumn", "1"))
  */
  // local multi-threaded Spark with HDFS using large dataset
  // not a good build test.
  /*
    ItemSimilarityDriver.main(Array(
      "--input", "hdfs://occam4:54310/user/pat/xrsj/ratings_data.txt",
      "--output", "hdfs://occam4:54310/user/pat/xrsj/similarityMatrices/",
      "--master", "local[4]",
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1"))
  */

  test("ItemSimilarityDriver, non-full-spec CSV") {

    val InFile = TmpDir + "in-file.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
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

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd = mahoutCtx.parallelize(lines).saveAsTextFile(InFile)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1",
      "--writeAllDatasets"))

    // todo: these comparisons rely on a sort producing the same lines, which could possibly
    // fail since the sort is on value and these can be the same for all items in a vector
    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(crossSimilarityLines) should contain theSameElementsAs CrossSimilarityTokens
  }



  test("ItemSimilarityDriver TSV ") {

    val InFile = TmpDir + "in-file.tsv/"
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1\tpurchase\tiphone",
      "u1\tpurchase\tipad",
      "u2\tpurchase\tnexus",
      "u2\tpurchase\tgalaxy",
      "u3\tpurchase\tsurface",
      "u4\tpurchase\tiphone",
      "u4\tpurchase\tgalaxy",
      "u1\tview\tiphone",
      "u1\tview\tipad",
      "u1\tview\tnexus",
      "u1\tview\tgalaxy",
      "u2\tview\tiphone",
      "u2\tview\tipad",
      "u2\tview\tnexus",
      "u2\tview\tgalaxy",
      "u3\tview\tsurface",
      "u3\tview\tnexus",
      "u4\tview\tiphone",
      "u4\tview\tipad",
      "u4\tview\tgalaxy")

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd = mahoutCtx.parallelize(lines).saveAsTextFile(InFile)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", "[,\t]",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1"))

    // todo: a better test would be to get sorted vectors and compare rows instead of tokens, this might miss
    // some error cases
    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(crossSimilarityLines) should contain theSameElementsAs CrossSimilarityTokens

  }

  test("ItemSimilarityDriver log-ish files") {

    val InFile = TmpDir + "in-file.log/"
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "2014-06-23 14:46:53.115\tu1\tpurchase\trandom text\tiphone",
      "2014-06-23 14:46:53.115\tu1\tpurchase\trandom text\tipad",
      "2014-06-23 14:46:53.115\tu2\tpurchase\trandom text\tnexus",
      "2014-06-23 14:46:53.115\tu2\tpurchase\trandom text\tgalaxy",
      "2014-06-23 14:46:53.115\tu3\tpurchase\trandom text\tsurface",
      "2014-06-23 14:46:53.115\tu4\tpurchase\trandom text\tiphone",
      "2014-06-23 14:46:53.115\tu4\tpurchase\trandom text\tgalaxy",
      "2014-06-23 14:46:53.115\tu1\tview\trandom text\tiphone",
      "2014-06-23 14:46:53.115\tu1\tview\trandom text\tipad",
      "2014-06-23 14:46:53.115\tu1\tview\trandom text\tnexus",
      "2014-06-23 14:46:53.115\tu1\tview\trandom text\tgalaxy",
      "2014-06-23 14:46:53.115\tu2\tview\trandom text\tiphone",
      "2014-06-23 14:46:53.115\tu2\tview\trandom text\tipad",
      "2014-06-23 14:46:53.115\tu2\tview\trandom text\tnexus",
      "2014-06-23 14:46:53.115\tu2\tview\trandom text\tgalaxy",
      "2014-06-23 14:46:53.115\tu3\tview\trandom text\tsurface",
      "2014-06-23 14:46:53.115\tu3\tview\trandom text\tnexus",
      "2014-06-23 14:46:53.115\tu4\tview\trandom text\tiphone",
      "2014-06-23 14:46:53.115\tu4\tview\trandom text\tipad",
      "2014-06-23 14:46:53.115\tu4\tview\trandom text\tgalaxy")

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd = mahoutCtx.parallelize(lines).saveAsTextFile(InFile)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", "\t",
      "--itemIDColumn", "4",
      "--rowIDColumn", "1",
      "--filterColumn", "2"))


    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(crossSimilarityLines) should contain theSameElementsAs CrossSimilarityTokens

  }

  test("ItemSimilarityDriver legacy supported file format") {

    val InDir = TmpDir + "in-dir/"
    val InFilename = "in-file.tsv"
    val InPath = InDir + InFilename

    val OutPath = TmpDir + "similarity-matrices"

    val lines = Array(
      "0,0,1",
      "0,1,1",
      "1,2,1",
      "1,3,1",
      "2,4,1",
      "3,0,1",
      "3,3,1")

    val Answer = tokenize(Iterable(
      "0\t1:1.7260924347106847",
      "3\t2:1.7260924347106847",
      "1\t0:1.7260924347106847",
      "4",
      "2\t3:1.7260924347106847"))

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(lines).coalesce(1, shuffle = true).saveAsTextFile(InDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(InDir + "part-00000"), new Path(InPath))

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InPath,
      "--output", OutPath,
      "--master", masterUrl))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs Answer

  }

  test("ItemSimilarityDriver write search engine output") {

    val InDir = TmpDir + "in-dir/"
    val InFilename = "in-file.tsv"
    val InPath = InDir + InFilename

    val OutPath = TmpDir + "similarity-matrices"

    val lines = Array(
      "0,0,1",
      "0,1,1",
      "1,2,1",
      "1,3,1",
      "2,4,1",
      "3,0,1",
      "3,3,1")

    val Answer = tokenize(Iterable(
      "0\t1",
      "3\t2",
      "1\t0",
      "4",
      "2\t3"))

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(lines).coalesce(1, shuffle = true).saveAsTextFile(InDir)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(InDir + "part-00000"), new Path(InPath))

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InPath,
      "--output", OutPath,
      "--master", masterUrl,
      "--omitStrength"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs Answer

  }

  test("ItemSimilarityDriver recursive file discovery using filename patterns") {
    //directory structure using the following
    // tmp/data/m1.tsv
    // tmp/data/more-data/another-dir/m2.tsv
    val M1Lines = Array(
      "u1\tpurchase\tiphone",
      "u1\tpurchase\tipad",
      "u2\tpurchase\tnexus",
      "u2\tpurchase\tgalaxy",
      "u3\tpurchase\tsurface",
      "u4\tpurchase\tiphone",
      "u4\tpurchase\tgalaxy")

    val M2Lines = Array(
      "u1\tview\tiphone",
      "u1\tview\tipad",
      "u1\tview\tnexus",
      "u1\tview\tgalaxy",
      "u2\tview\tiphone",
      "u2\tview\tipad",
      "u2\tview\tnexus",
      "u2\tview\tgalaxy",
      "u3\tview\tsurface",
      "u3\tview\tnexus",
      "u4\tview\tiphone",
      "u4\tview\tipad",
      "u4\tview\tgalaxy")

    val InFilenameM1 = "m1.tsv"
    val InDirM1 = TmpDir + "data/"
    val InPathM1 = InDirM1 + InFilenameM1
    val InFilenameM2 = "m2.tsv"
    val InDirM2 = TmpDir + "data/more-data/another-dir/"
    val InPathM2 = InDirM2 + InFilenameM2

    val InPathStart = TmpDir + "data/"
    val OutPath = TmpDir + "similarity-matrices"

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(M1Lines).coalesce(1, shuffle = true).saveAsTextFile(InDirM1)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    val fs = FileSystem.get(new Configuration())
    //rename part-00000 to something.tsv
    fs.rename(new Path(InDirM1 + "part-00000"), new Path(InPathM1))

    // this creates one part-0000 file in the directory
    mahoutCtx.parallelize(M2Lines).coalesce(1, shuffle = true).saveAsTextFile(InDirM2)

    // to change from using part files to a single .tsv file we'll need to use HDFS
    //rename part-00000 to tmp/some-location/something.tsv
    fs.rename(new Path(InDirM2 + "part-00000"), new Path(InPathM2))

    // local multi-threaded Spark with default FS, suitable for build tests but need better location for data

    ItemSimilarityDriver.main(Array(
      "--input", InPathStart,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", "\t",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1",
      "--filenamePattern", "m..tsv",
      "--recursive"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(crossSimilarityLines) should contain theSameElementsAs CrossSimilarityTokens

  }

  test("ItemSimilarityDriver, two input paths") {

    val InFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val InFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
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

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile1)
    val linesRdd2 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile2)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile1,
      "--input2", InFile2,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(crossSimilarityLines) should contain theSameElementsAs CrossSimilarityTokens

  }

  test("ItemSimilarityDriver, two inputs of different dimensions") {

    val InFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val InFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      // remove one user so A'B will be of different dimensions
      // ItemSimilarityDriver should create one unified user dictionary and so account for this
      // discrepancy as a blank row: "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
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

    val UnequalDimensionsSelfSimilarity = tokenize(Iterable(
      "ipad\tiphone:1.7260924347106847",
      "iphone\tipad:1.7260924347106847",
      "nexus\tgalaxy:1.7260924347106847",
      "galaxy\tnexus:1.7260924347106847"))

    //only surface purchase was removed so no cross-similarity for surface
    val UnequalDimensionsCrossSimilarity = tokenize(Iterable(
      "galaxy\tgalaxy:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 nexus:1.7260924347106847",
      "iphone\tgalaxy:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 nexus:1.7260924347106847",
      "ipad\tgalaxy:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 nexus:0.6795961471815897",
      "nexus\tiphone:0.6795961471815897 ipad:0.6795961471815897 nexus:0.6795961471815897 galaxy:0.6795961471815897"))
    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile1)
    val linesRdd2 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile2)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile1,
      "--input2", InFile2,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs UnequalDimensionsSelfSimilarity
    tokenize(crossSimilarityLines) should contain theSameElementsAs UnequalDimensionsCrossSimilarity

  }

  test("ItemSimilarityDriver cross similarity two separate items spaces") {
    /* cross-similarity with category views, same user space
            	phones	tablets	mobile_acc	soap
          u1	0	      1	      1	          0
          u2	1	      1	      1	          0
          u3	0	      0	      1	          0
          u4	1	      1	      0	          1
    */
    val InFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val InFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
      "u1,view,phones",
      "u1,view,mobile_acc",
      "u2,view,phones",
      "u2,view,tablets",
      "u2,view,mobile_acc",
      "u3,view,mobile_acc",
      "u4,view,phones",
      "u4,view,tablets",
      "u4,view,soap")

    val UnequalDimensionsCrossSimilarityLines = tokenize(Iterable(
      "iphone\tmobile_acc:1.7260924347106847 soap:1.7260924347106847 phones:1.7260924347106847",
      "surface\tmobile_acc:0.6795961471815897",
      "nexus\ttablets:1.7260924347106847 mobile_acc:0.6795961471815897 phones:0.6795961471815897",
      "galaxy\ttablets:5.545177444479561 soap:1.7260924347106847 phones:1.7260924347106847 " +
        "mobile_acc:1.7260924347106847",
      "ipad\tmobile_acc:0.6795961471815897 phones:0.6795961471815897"))

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile1)
    val linesRdd2 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile2)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile1,
      "--input2", InFile2,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1",
      "--writeAllDatasets"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    tokenize(crossSimilarityLines) should contain theSameElementsAs UnequalDimensionsCrossSimilarityLines

  }

  test("A.t %*% B after changing row cardinality of A") {
    // todo: move to math tests but this is Spark specific

    val a = dense(
      (1.0, 1.0))

    val b = dense(
      (1.0, 1.0),
      (1.0, 1.0),
      (1.0, 1.0))

    val inCoreABiggertBAnswer = dense(
      (1.0, 1.0),
      (1.0, 1.0))

    val drmA = drmParallelize(m = a, numPartitions = 2)
    val drmB = drmParallelize(m = b, numPartitions = 2)

    // modified to return a new CheckpointedDrm so maintains immutability but still only increases the row cardinality
    // by returning new CheckpointedDrmSpark[K](rdd, n, ncol, _cacheStorageLevel ) Hack for now.
    val drmABigger = drmWrap[Int](drmA.rdd, 3, 2)


    val ABiggertB = drmABigger.t %*% drmB
    val inCoreABiggertB = ABiggertB.collect

    assert(inCoreABiggertB === inCoreABiggertBAnswer)

    val bp = 0
  }

  test("Changing row cardinality of an IndexedDataset") {

    val a = dense(
      (1.0, 1.0))

    val drmA = drmParallelize(m = a, numPartitions = 2)
    val emptyIDs = new BiDictionary(new HashMap[String, Int]())
    val indexedDatasetA = new IndexedDatasetSpark(drmA, emptyIDs, emptyIDs)
    val biggerIDSA = indexedDatasetA.newRowCardinality(5)

    assert(biggerIDSA.matrix.nrow == 5)

  }

  test("ItemSimilarityDriver cross similarity two separate items spaces, missing rows in B") {
    /* cross-similarity with category views, same user space
            	phones	tablets	mobile_acc	soap
            u1	0	      1	      1	          0
            u2	1	      1	      1	          0
removed ==> u3	0	      0	      1	          0
            u4	1	      1	      0	          1
    */
    val InFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val InFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
      "u1,view,phones",
      "u1,view,mobile_acc",
      "u2,view,phones",
      "u2,view,tablets",
      "u2,view,mobile_acc",
      //"u3,view,mobile_acc",// if this line is removed the cross-cooccurrence should work
      "u4,view,phones",
      "u4,view,tablets",
      "u4,view,soap")

    val UnequalDimensionsCrossSimilarityLines = tokenize(Iterable(
      "galaxy\ttablets:5.545177444479561 soap:1.7260924347106847 phones:1.7260924347106847",
      "ipad\tmobile_acc:1.7260924347106847 phones:0.6795961471815897",
      "surface",
      "nexus\tmobile_acc:1.7260924347106847 tablets:1.7260924347106847 phones:0.6795961471815897",
      "iphone\tsoap:1.7260924347106847 phones:1.7260924347106847"))

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile1)
    val linesRdd2 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile2)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile1,
      "--input2", InFile2,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1",
      "--writeAllDatasets"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs SelfSimilairtyTokens
    tokenize(crossSimilarityLines) should contain theSameElementsAs UnequalDimensionsCrossSimilarityLines
  }

  test("ItemSimilarityDriver cross similarity two separate items spaces, adding rows in B") {
    /* cross-similarity with category views, same user space
            	phones	tablets	mobile_acc	soap
            u1	0	      1	      1	          0
            u2	1	      1	      1	          0
removed ==> u3	0	      0	      1	          0
            u4	1	      1	      0	          1
    */
    val InFile1 = TmpDir + "in-file1.csv/" //using part files, not single file
    val InFile2 = TmpDir + "in-file2.csv/" //using part files, not single file
    val OutPath = TmpDir + "similarity-matrices/"

    val lines = Array(
      "u1,purchase,iphone",
      "u1,purchase,ipad",
      "u2,purchase,nexus",
      "u2,purchase,galaxy",
      "u3,purchase,surface",
      "u4,purchase,iphone",
      "u4,purchase,galaxy",
      "u1,view,phones",
      "u1,view,mobile_acc",
      "u2,view,phones",
      "u2,view,tablets",
      "u2,view,mobile_acc",
      "u3,view,mobile_acc",// if this line is removed the cross-cooccurrence should work
      "u4,view,phones",
      "u4,view,tablets",
      "u4,view,soap",
      "u5,view,soap")

    val UnequalDimensionsSimilarityTokens = List(
      "galaxy",
      "nexus:2.231435513142097",
      "iphone:0.13844293808390518",
      "nexus",
      "galaxy:2.231435513142097",
      "ipad",
      "iphone:2.231435513142097",
      "surface",
      "iphone",
      "ipad:2.231435513142097",
      "galaxy:0.13844293808390518")

    val UnequalDimensionsCrossSimilarityLines = List(
      "galaxy",
      "tablets:6.730116670092563",
      "phones:2.9110316603236868",
      "soap:0.13844293808390518",
      "mobile_acc:0.13844293808390518",
      "nexus",
      "tablets:2.231435513142097",
      "mobile_acc:1.184939225613002",
      "phones:1.184939225613002",
      "ipad", "mobile_acc:1.184939225613002",
      "phones:1.184939225613002",
      "surface",
      "mobile_acc:1.184939225613002",
      "iphone",
      "phones:2.9110316603236868",
      "soap:0.13844293808390518",
      "tablets:0.13844293808390518",
      "mobile_acc:0.13844293808390518")

    // this will create multiple part-xxxxx files in the InFile dir but other tests will
    // take account of one actual file
    val linesRdd1 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile1)
    val linesRdd2 = mahoutCtx.parallelize(lines).saveAsTextFile(InFile2)

    // local multi-threaded Spark with default HDFS
    ItemSimilarityDriver.main(Array(
      "--input", InFile1,
      "--input2", InFile2,
      "--output", OutPath,
      "--master", masterUrl,
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDColumn", "2",
      "--rowIDColumn", "0",
      "--filterColumn", "1",
      "--writeAllDatasets"))

    val similarityLines = mahoutCtx.textFile(OutPath + "/similarity-matrix/").collect.toIterable
    val crossSimilarityLines = mahoutCtx.textFile(OutPath + "/cross-similarity-matrix/").collect.toIterable
    tokenize(similarityLines) should contain theSameElementsAs UnequalDimensionsSimilarityTokens
    tokenize(crossSimilarityLines) should contain theSameElementsAs UnequalDimensionsCrossSimilarityLines
  }

  // convert into an Iterable of tokens for 'should contain theSameElementsAs Iterable'
  def tokenize(a: Iterable[String]): Iterable[String] = {
    var r: Iterable[String] = Iterable()
    a.foreach { l =>
      l.split("\t").foreach { s =>
        r = r ++ s.split("[\t ]")
      }
    }
    r
  }

  override protected def beforeAll(configMap: ConfigMap) {
    super.beforeAll(configMap)
    ItemSimilarityDriver.useContext(mahoutCtx)
  }

}
