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

import org.apache.mahout.math.indexeddataset.DefaultIndexedDatasetReadSchema
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.FunSuite

import scala.collection.JavaConversions._

class TextDelimitedReaderWriterSuite extends FunSuite with DistributedSparkSuite {
  test("indexedDatasetDFSRead should read sparse matrix file with null rows") {
    val OutFile = TmpDir + "similarity-matrices/part-00000"

    val lines = Array(
      "galaxy\tnexus:1.0",
      "ipad\tiphone:2.0",
      "nexus\tgalaxy:3.0",
      "iphone\tipad:4.0",
      "surface"
    )
    val linesRdd = mahoutCtx.parallelize(lines).saveAsTextFile(OutFile)

    val data = mahoutCtx.indexedDatasetDFSRead(OutFile, DefaultIndexedDatasetReadSchema)

    data.rowIDs.toMap.keySet should equal(Set("galaxy", "ipad", "nexus", "iphone", "surface"))
    data.columnIDs.toMap.keySet should equal(Set("nexus", "iphone", "galaxy", "ipad"))

    val a = data.matrix.collect
    a.setRowLabelBindings(mapAsJavaMap(data.rowIDs.toMap).asInstanceOf[java.util.Map[java.lang.String, java.lang.Integer]])
    a.setColumnLabelBindings(mapAsJavaMap(data.columnIDs.toMap).asInstanceOf[java.util.Map[java.lang.String, java.lang.Integer]])
    a.get("galaxy", "nexus") should equal(1.0)
    a.get("ipad", "iphone") should equal(2.0)
    a.get("nexus", "galaxy") should equal(3.0)
    a.get("iphone", "ipad") should equal(4.0)
  }
}
