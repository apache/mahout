/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */

package org.apache.mahout.spark.sparkbindings.algorithms

import org.apache.spark.ml.param._

/**
 * These are shared parameters used for the Spark models.
 * For the ones already defined in Spark, normally we would extend these from Spark, but they
 * are currently private. Instead we define our own. In the future these
 * will be code generated.
 */

/**
 * Trait for shared param inputCol.
 */
trait HasInputCol extends Params {

  /**
   * Param for input column name.
   * @group param
   */
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")

  /** @group getParam */
  final def getInputCol: String = $(inputCol)

  final def setInputCol(value: String): this.type = set(this.inputCol, value)
}

/**
 * Trait for shared param outputCol (default: uid + "__output").
 */
trait HasOutputCol extends Params {

  /**
   * Param for output column name.
   * @group param
   */
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")

  setDefault(outputCol, uid + "__output")

  /** @group getParam */
  final def getOutputCol: String = $(outputCol)

  final def setOutputCol(value: String): this.type = set(this.outputCol, value)
}

/**
 * Trait for shared param featuresCol (default: "features").
 */
trait HasFeaturesCol extends Params {

  /**
   * Param for features column name.
   * @group param
   */
  final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "features column name")

  setDefault(featuresCol, "features")

  /** @group getParam */
  final def getFeaturesCol: String = $(featuresCol)

  final def setFeaturesCol(value: String): this.type = set(this.featuresCol, value)
}

/**
 * Trait for shared param labelCol (default: "label").
 */
trait HasLabelCol extends Params {

  /**
   * Param for label column name.
   * @group param
   */
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

  setDefault(labelCol, "label")

  /** @group getParam */
  final def getLabelCol: String = $(labelCol)

  final def setLabelCol(value: String): this.type = set(this.labelCol, value)
}
