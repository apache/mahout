package org.apache.mahout.cylon-example.flinkengine.schemas

import java.awt.image.BufferedImage

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala.createTypeInformation
import org.apache.flink.streaming.util.serialization.KeyedDeserializationSchema
import org.apache.mahout.math.Vector
import org.apache.mahout.math.scalabindings.MahoutCollections._
import org.apache.mahout.cylon-example.common.mahout.MahoutUtils
import org.slf4j.{Logger, LoggerFactory}


import org.apache.mahout.math.{DenseVector, Matrix, Vector}
import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.MahoutCollections._


// Schema: (topic: String, x: Int, y: Int, v: MahoutVector)
class MahoutVectorAndCoordsSchema extends KeyedDeserializationSchema[(String, Int, Int, Int, Int, Int, Vector)]
  //with KeyedSerializationSchema[(String, BufferedImage)]  // Don't think we need this
{

  val logger: Logger = LoggerFactory.getLogger(classOf[MahoutVectorAndCoordsSchema])

  def isEndOfStream(nextElement: (String, Int, Int, Int, Int, Int, Vector)): Boolean = {
    false
  }

  def getProducedType: TypeInformation[(String, Int, Int, Int, Int, Int, Vector)] =
    createTypeInformation[(String, Int, Int, Int, Int, Int, Vector)]

  def deserialize(messageKey: Array[Byte],
                  message: Array[Byte],
                  topic: String,
                  partition: Int,
                  offset: Long): (String, Int, Int, Int, Int, Int, Vector) = {
    val v = MahoutUtils.byteArray2vector(message)
    val h = v.get(0).toInt
    val w = v.get(1).toInt
    val x = v.get(2).toInt
    val y = v.get(3).toInt
    val frame = v.get(4).toInt
    val outputV = dvec(v.toArray.slice(2, v.size()))
    (new String(messageKey, "UTF-8"), h, w, x, y, frame, outputV)
  }
}
