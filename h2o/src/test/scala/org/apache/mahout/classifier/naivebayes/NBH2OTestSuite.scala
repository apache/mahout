package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.h2obindings.test.DistributedH2OSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.FunSuite

class NBH2OTestSuite extends FunSuite with DistributedH2OSuite with MahoutSuite with NBTestBase
