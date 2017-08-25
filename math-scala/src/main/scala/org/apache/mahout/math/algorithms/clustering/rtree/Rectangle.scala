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


package org.apache.mahout.math.algorithms.clustering.rtree

import scala.math._

/*
  A rectangle is defined by two parameters. The coordinates of its topright corner and the coordinates of its bootomleftcorner
 */
trait Rectangle{
  
  require(bottomLeft.size == topRight.size) //Both the corners should have the same ns

  def bottomLeft: List[Double] //Coordinates of bottom left corner
  def topRight: List[Double] //Coordinates of top right corner
  def n: Int = bottomLeft.size //nality

  //Return the lowerLeft corner as a Point
  def lowerLeftPoint: Point = Point(this.bottomLeft)

  //Return the upper right corner as a Point
  def upperRightPoint: Point = Point(this.topRight)

  //Checks if this contains the Rectangle rect
  def contains(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.n ) {
      if( !(this.bottomLeft(i) <= rect.bottomLeft(i) && rect.topRight(i) <= this.topRight(i)) ) {
        false
      }
    }
    true
  }

  def wraps(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.n ) {
      if( !(this.bottomLeft(i) < rect.bottomLeft(i) && rect.topRight(i) < this.topRight(i)) ) {
        false
      }
    }
    true
  }

  def intersects(rect: Rectangle): Boolean = {
    for(i <- 0 until rect.n) {
      if( !(this.bottomLeft(i) <= rect.topRight(i) && rect.bottomLeft(i) <= this.topRight(i)) )
        false
    }
    true
  }

  def area(): Double = {
    var area = 1.0
    for(i <- 0 until this.n) {
      area = area * (this.topRight(i) - this.bottomLeft(i))
    }
    area
  }

  def toMBR: MBR = MBR(this.bottomLeft, this.topRight)

  def expandRectangle(rect: Rectangle): MBR = {
    val minBottomLeft = new Array[Double](rect.n)
    val maxTopRight = new Array[Double](rect.n)
    for(i <- 0 until rect.n) {
      minBottomLeft(i) = math.min(bottomLeft(i), rect.bottomLeft(i))
      maxTopRight(i) = math.max(topRight(i), rect.topRight(i))
    }
    MBR(minBottomLeft.toList, maxTopRight.toList)
  }

  //Area of 'this' rectangle after merging with 'rect'
  def diffAreaAfterExpansion(rect: Rectangle): Double = {
    val minBottomLeft = new Array[Double](rect.n)
    val maxTopRight = new Array[Double](rect.n)
    var area: Double = 0

    for(i <- 0 until rect.n) {
      minBottomLeft(i) = math.min(bottomLeft(i), rect.bottomLeft(i))
      maxTopRight(i) = math.max(topRight(i), rect.topRight(i))
      area += (maxTopRight(i) - minBottomLeft(i))*(maxTopRight(i) - minBottomLeft(i))
    }
    area - this.area
  }

  //Distance of a point from the rectangle.
  def distance(pt: Point): Double = {
    val n = pt.coordinates.size
    val dList = new Array[Double](n)
    for(dim <- 0 until n) {
      var x = this.bottomLeft(dim)
      var x2 = this.topRight(dim)
      dList(dim) = if (pt.coordinates(dim) < x) x - pt.coordinates(dim) else if (pt.coordinates(dim) < x2) 0D else pt.coordinates(dim) - x2
    }
    var squaredSum: Double = 0
    for(i <- 0 until n) {
      squaredSum += dList(i)*dList(i)
    }
    sqrt(squaredSum)
  }

}

//Represents a point with coordinates and n information
case class Point(coordinates: List[Double]) extends Rectangle {

  override def bottomLeft: List[Double] = coordinates
  override def topRight: List[Double] = coordinates

  override def n: Int = bottomLeft.size

  override def lowerLeftPoint: Point = this
  override def upperRightPoint: Point = this

  override def distance(pt: Point): Double = {
    var squaredSum: Double = 0
    for(i <- 0 until pt.n){
      squaredSum += (pt.bottomLeft(i) - bottomLeft(i))*(pt.bottomLeft(i) - bottomLeft(i))
    }
    squaredSum
  }

  override def wraps(rect: Rectangle): Boolean = false
}

//MBR is a physical manifestation of the Rectangle trait
case class MBR(bottomLeft: List[Double], topRight: List[Double]) extends Rectangle {
  override def toMBR: MBR = this
}

object MBR {

  val empty: MBR = {
    val s: List[Double] = List.fill(10)(Math.sqrt(Double.MaxValue))
    val t: List[Double] = List.fill(10)(s(0) + -2.0F* s(0))
    MBR(s, t)

  }

}