package org.apache.mahout.drivers

import org.scalatest.{BeforeAndAfter, FunSuite}

import org.apache.mahout.sparkbindings._
import java.io.{FileWriter, BufferedWriter}
import com.google.common.io.Closeables
import org.apache.spark.SparkContext

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

class ItemSimilarityDriver$Test extends FunSuite {
  private var sc: SparkContext = _


  test ("Running some text delimited files through the import/cooccurrence/export"){

    val exampleCsvlogStatements = Array(
      "12569537329,user1,item1,\"view\"",
      "12569537329,user1,item2,\"view\"",
      "12569537329,user1,item2,\"like\"",
      "12569537329,user1,item3,\"like\"",
      "12569537329,user2,item2,\"view\"",
      "12569537329,user2,item2,\"like\"",
      "12569537329,user3,item1,\"like\"",
      "12569537329,user3,item3,\"view\"",
      "12569537329,user3,item3,\"like\""
    )

    val exampleTsvLogStatements = Array(
      "12569537329\tuser1\titem1\t\"view\"",
      "12569537329\tuser1\titem2\t\"view\"",
      "12569537329\tuser1\titem2\t\"like\"",
      "12569537329\tuser1\titem3\t\"like\"",
      "12569537329\tuser2\titem2\t\"view\"",
      "12569537329\tuser2\titem2\t\"like\"",
      "12569537329\tuser3\titem1\t\"like\"",
      "12569537329\tuser3\titem3\t\"view\"",
      "12569537329\tuser3\titem3\t\"like\""
    )


    val csvLogStatements = Array(
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
      "u4,view,galaxy"
    )

    val tsvLogStatements = Array(
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
      "u4\tview\tgalaxy"
    )

    var w: BufferedWriter = null
    //try {
      w = new BufferedWriter(new FileWriter("/tmp/cf-data.txt"))
      w.write(csvLogStatements.mkString("\n"))
    //} finally {
      Closeables.close(w, false)
    //}
    /*
        val indexedLikes = IndexedDatasetStore.readTuples(sc, "tmp/cf-data.txt", 2, ",", 0, 2, 1, "purchase")

        val indexedViews = IndexedDatasetStore.readTuples(sc, "/tmp/cf-data.txt", 2, ",", 0, 2, 1, "view")

        val drmLikes = indexedLikes.matrix
        val drmViews = indexedViews.matrix

        // Now we could run cooccurrence analysis using the DRMs, instead we'll just fetch and print the matrices
        val drmXCooccurrences = cooccurrences(drmLikes, randomSeed = 0xdeadbeef,
          maxInterestingItemsPerThing = 100, maxNumInteractions = 500, Array(drmViews))

        val inCoreViews = drmViews.collect
        val inCoreLikes = drmLikes.collect
        val inCoreIndicator = drmXCooccurrences(0).collect
        val inCoreXIndicator = drmXCooccurrences(1).collect
        println("\nLIKES:")
        println(inCoreLikes)
        println("\nVIEWS:")
        println(inCoreViews)
        println("\nINDICATOR MATRIX")
        println(inCoreIndicator)
        println("\nCROSS INDICATOR MATRIX")
        println(inCoreXIndicator)
    */

    /*
        //Clustered Spark and HDFS
        ItemSimilarityDriver.main(Array(
          "--input", "hdfs://occam4:54310/user/pat/spark-itemsimilarity/cf-data.txt",
          "--output", "hdfs://occam4:54310/user/pat/spark-itemsimilarity/indicatorMatrices/",
          "--master", "spark://occam4:7077",
          "--filter1", "purchase",
          "--filter2", "view",
          "--inDelim", ",",
          "--itemIDPosition", "2",
          "--rowIDPosition", "0",
          "--filterPosition", "1"
        ))
    */
    //local multi-threaded Spark with HDFS using large dataset
    /*    ItemSimilarityDriver.main(Array(
        "--input", "hdfs://occam4:54310/user/pat/xrsj/ratings_data.txt",
        "--output", "hdfs://occam4:54310/user/pat/xrsj/indicatorMatrices/",
        "--master", "local[4]",
        "--filter1", "purchase",
        "--filter2", "view",
        "--inDelim", ",",
        "--itemIDPosition", "2",
        "--rowIDPosition", "0",
        "--filterPosition", "1"
      ))
    */

    //local multi-threaded Spark with local FS
    ItemSimilarityDriver.main(Array(
      "--input", "/tmp/cf-data.txt",
      "--output", "tmp/indicatorMatrices/",
      "--master", "local[4]",
      "--filter1", "purchase",
      "--filter2", "view",
      "--inDelim", ",",
      "--itemIDPosition", "2",
      "--rowIDPosition", "0",
      "--filterPosition", "1"
    ))

  }

}
