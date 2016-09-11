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

package org.apache.mahout.sparkbindings

import java.io.{Closeable, File}

import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.util.IOUtilsScala
import org.scalatest.FunSuite

import scala.collection._


class SparkBindingsSuite extends FunSuite with DistributedSparkSuite {

  // This test will succeed only when MAHOUT_HOME is set in the environment. So we keep it for
  // diagnostic purposes around, but we probably don't want it to run in the Jenkins, so we'd
  // let it to be ignored.
  ignore("context jars") {
    System.setProperty("mahout.home", new File("..").getAbsolutePath/*"/home/dmitriy/projects/github/mahout-commits"*/)
    val closeables = new mutable.ListBuffer[Closeable]()
    try {
      val mahoutJars = findMahoutContextJars(closeables)
      mahoutJars.foreach {
        println(_)
      }

      mahoutJars.size should be > 0
      // this will depend on the viennacl profile.
      // mahoutJars.size shouldBe 4
    } finally {
      IOUtilsScala.close(closeables)
    }

  }

}
