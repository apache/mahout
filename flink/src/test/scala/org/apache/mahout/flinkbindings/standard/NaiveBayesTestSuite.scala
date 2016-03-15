package org.apache.mahout.flinkbindings.standard

import org.apache.mahout.classifier.naivebayes.NBTestBase
import org.apache.mahout.flinkbindings._
import org.scalatest.FunSuite


class NaiveBayesTestSuite extends FunSuite with DistributedFlinkSuite
      with NBTestBase {

}