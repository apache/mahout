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

package org.apache.mahout.math.algorithms.clustering.RTree

import java.lang.Double.{isInfinite, isNaN}

import scala.math.sqrt

//Represents the basic constituent of an R-Tree. A Rectangle.
trait Rectangle{

  require(bottomLeft.size == topRight.size) //Both the corners should have the same dimensions

  def bottomLeft: List[Double] //Coordinates
  def topRight: List[Double] //Coordinates
  def dimension: Int = bottomLeft.size //Dimensionality
  def rectArea: Double = area(this)

  def squaredDistance(pt: Point): Double = {
    val dimension = pt.coordinates.size
    val dList = new Array[Double](dimension)
    for(dim <- 0 until dimension) {
      var x = this.bottomLeft(dim)
      var x2 = this.topRight(dim)
      dList(dim) = if (pt.coordinates(dim) < x) x - pt.coordinates(dim) else if (pt.coordinates(dim) < x2) 0D else pt.coordinates(dim) - x2
    }
    var squaredSum: Double = 0
    for(i <- 0 until dimension) {
      squaredSum += dList(i)*dList(i)
    }
    squaredSum
  }

  def distanceActual(pt: Point): Double =
    sqrt(squaredDistance(pt))

  def isValid: Boolean = {
    for (i <- 0 until dimension) {
      if (isNaN(this.bottomLeft(i)) || isNaN(this.topRight(i)) || isInfinite(this.bottomLeft(i)) || isInfinite((this.topRight(i)))) false
    }
    true
  }

  def toBox: Box = Box(this.bottomLeft, this.topRight)

  //Return the lowerLeft corner as a Point
  def lowerLeft: Point = Point(this.bottomLeft)

  //Return the upper right corner as a Point
  def upperRight: Point = Point(this.topRight)

  //Checks if this contains the Rectangle rect
  def contains(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.dimension ) {
      if( !(this.bottomLeft(i) <= rect.bottomLeft(i) && rect.topRight(i) <= this.topRight(i)) ) {
        false
      }
    }
    true
  }

  def wraps(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.dimension ) {
      if( !(this.bottomLeft(i) < rect.bottomLeft(i) && rect.topRight(i) < this.topRight(i)) ) {
        false
      }
    }
    true
  }

  def intersects(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.dimension) {
      if( !(this.bottomLeft(i) <= rect.topRight(i) && rect.bottomLeft(i) <= this.topRight(i)) )
        false
    }
    true
  }

  def area(rect: Rectangle): Double = {
    var area = 1.0
    for(i <- 0 until rect.dimension) {
      area = area * (rect.topRight(i) - rect.bottomLeft(i))
    }
    area
  }

  def expandRectangle(rect: Rectangle): Box = {
    val minBottomLeft = new Array[Double](rect.dimension)
    val maxTopRight = new Array[Double](rect.dimension)
    for(i <- 0 until rect.dimension) {
      minBottomLeft(i) = math.min(bottomLeft(i), rect.bottomLeft(i))
      maxTopRight(i) = math.max(topRight(i), rect.topRight(i))
    }

    Box(minBottomLeft.toList, maxTopRight.toList)
  }

  def expandArea(rect: Rectangle): Double = {
    val minBottomLeft = new Array[Double](rect.dimension)
    val maxTopRight = new Array[Double](rect.dimension)
    var area: Double = 0

    for(i <- 0 until rect.dimension) {
      minBottomLeft(i) = math.min(bottomLeft(i), rect.bottomLeft(i))
      maxTopRight(i) = math.max(topRight(i), rect.topRight(i))
      area += (maxTopRight(i) - minBottomLeft(i))*(maxTopRight(i) - minBottomLeft(i))
    }

    area - this.area(this)
  }

}

//Represents a point with coordinates and dimension information
case class Point(coordinates: List[Double]) extends Rectangle {

  override def bottomLeft: List[Double] = coordinates
  override def topRight: List[Double] = coordinates

  override def dimension: Int = bottomLeft.size

  override def lowerLeft: Point = this
  override def upperRight: Point = this

  override def squaredDistance(pt: Point): Double = {
    var squaredSum: Double = 0
    for(i <- 0 until pt.dimension){
      squaredSum += (pt.bottomLeft(i) - bottomLeft(i))*(pt.bottomLeft(i) - bottomLeft(i))
    }
    squaredSum
  }

  override def isValid: Boolean = {
    for (i <- 0 until dimension) {
      if (isNaN(bottomLeft(i)) || isInfinite(bottomLeft(i))) {
        false
      }
    }
    true
  }
  override def wraps(rect: Rectangle): Boolean = false
}

//Box is a physical manifestation of the Rectangle trait
case class Box(bottomLeft: List[Double], topRight: List[Double]) extends Rectangle {
  override def toBox: Box = this
}

object Box {

  val empty: Box = {
    val s: List[Double] = List.fill(10)(Math.sqrt(Double.MaxValue))
    val t: List[Double] = List.fill(10)(s(0) + -2.0F* s(0))
    Box(s, t)

  }

}
