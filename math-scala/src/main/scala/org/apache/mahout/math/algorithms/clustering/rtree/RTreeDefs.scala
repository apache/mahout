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

import scala.collection.mutable.ArrayBuffer

object Constants{
  def Max = 10
  def Min = 5
}

abstract class Rect{
  def rect: Rectangle
}

abstract class Node[K] extends Rect
{
  def mbr: MBR
  def rect: Rectangle = mbr
  def internal: Boolean
  def external: Boolean

  def children: Vector[Node[K]]
  def nodeMembers: Vector[Node[K]] = {
    val buf = ArrayBuffer.empty[Node[K]]
    buf.toVector
  }

  def returnSubTree: Vector[Node[K]] = {
    var list: ArrayBuffer[Node[K]] = ArrayBuffer.empty

    def helper(node: Node[K]): ArrayBuffer[Node[K]] = {
      if(node.external) {
       node.children.toBuffer.asInstanceOf[ArrayBuffer[Node[K]]]
      }
      else{
        for(i <- 0 until children.size){
          list ++= helper(children(i))
        }
        list
      }
    }
    list.toVector
  }

}

case class InternalNode[K](children: Vector[Node[K]], mbr: MBR) extends Node[K]{
  override def internal: Boolean = true

  override def external: Boolean = false


}

case class ExternalNode[K](children: Vector[Node[K]], mbr: MBR) extends Node[K]{

  override def internal: Boolean = false

  override def external: Boolean = true

}

class RTreeDefs {



}
