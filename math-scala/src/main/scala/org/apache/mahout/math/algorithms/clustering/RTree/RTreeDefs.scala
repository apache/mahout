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

import scala.collection.mutable.{ArrayBuffer, PriorityQueue}

object Constants {
  // $COVERAGE-OFF$
  @inline final val MaxEntries = 50
  // $COVERAGE-ON$
}

import org.apache.mahout.math.algorithms.clustering.RTree.Constants._

sealed abstract class HasRectangle {
  def rect: Rectangle
}

//Contains code related to the definitions of an R-Tree
sealed abstract class Node[A] extends HasRectangle { self =>

  def box: Box
  def rect: Rectangle = box

  def children: Vector[HasRectangle]

  def entries: Vector[Entry[A]] = {
    val buf = ArrayBuffer.empty[Entry[A]]

    def recur(node: Node[A]): Unit = node match {
      case External(children, _) =>
        buf ++= children
      case Internal(children, _) =>
        children.foreach(recur)
    }
    recur(this)
    buf.toVector
  }

  def iterator: Iterator[Entry[A]] = this match {
    case External(children, _) =>
      children.iterator
    case Internal(children, _) =>
      children.iterator.flatMap(_.iterator)
  }

//  def pretty: String = {
//    def prettyRecur(node: Node[A], i: Int, sb: StringBuilder): Unit = {
//      val pad = " " * i
//      val a = node.box.area(node.box)
//      node match {
//        case lf @ External(children, box) =>
//          val pad2 = " " * (i + 1)
//          sb.append(s"$pad leaf $a $box:\n")
//          children.foreach { case Entry(pt, value) =>
//            sb.append(s"$pad2 entry $pt: $value\n")
//          }
//        case Internal(children, box) =>
//          sb.append(s"$pad branch $a $box:\n")
//          children.foreach(c => prettyRecur(c, i + 1, sb))
//      }
//    }
//    val sb = new StringBuilder
//    prettyRecur(this, 0, sb)
//    sb.toString
//  }

  def insert(entry: Entry[A]): Either[Vector[Node[A]], Node[A]] = {
    this match {
      case External(children, box) =>
        val cs = children :+ entry
        if (cs.length <= MaxEntries) {
          Right(External(cs, box.expandRectangle(entry.rect)))
        } else {
          Left(Node.splitExternal(cs))
        }

      case Internal(children, box) =>
        assert(children.length > 0)

        // here we need to find the "best" child to put the entry
        // into. we define that as the child that needs to add the
        // least amount of area to its own bounding box to accomodate
        // the new entry.
        //
        // the results are "node", the node to add to, and "n", the
        // position of that node in our vector.
        val pt = entry.rect
        var node = children(0)
        var n = 0
        var area = node.box.expandArea(pt)
        var i = 1
        while (i < children.length) {
          val curr = children(i)
          val a = curr.box.expandArea(pt)
          if (a < area) {
            area = a
            n = i
            node = curr
          }
          i += 1
        }

        // now we perform the actual insertion into the node. as
        // stated above, that node will either return a single new
        // node (Right) or a vector of new nodes (Left).
        node.insert(entry) match {
          case Left(rs) =>
            val cs = children.take(n) ++ children.drop(n + 1) ++ rs
            if (cs.length <= MaxEntries) {
              val b = rs.foldLeft(box)(_ expandRectangle _.box)
              Right(Internal(cs, b))
            } else {
              Left(Node.splitInternal(cs))
            }
          case Right(r) =>
            val cs = children.updated(n, r)
            if (cs.length <= MaxEntries) {
              Right(Internal(children.updated(n, r), box.expandRectangle(r.box)))
            } else {
              Left(Node.splitInternal(cs))
            }
        }
    }
  }

  /**
    * Determine if we need to try contracting our bounding box based on
    * the loss of 'rect'. If so, use the by-name parameter 'regen' to
    * recalculate. Since regen is by-name, it won't be evaluated unless
    * we need it.
    */
  def contract(gone: Rectangle, regen: => Box): Box =
    if (box.wraps(gone)) box else regen

  def remove(entry: Entry[A]): Option[(ADT[Entry[A]], Option[Node[A]])]

  def search(space: Box, f: Entry[A] => Boolean): Seq[Entry[A]] =
    genericSearch(space, space.contains, f)

  def searchIntersection(space: Box, f: Entry[A] => Boolean): Seq[Entry[A]] =
    genericSearch(space, space.intersects, f)

  def genericSearch(space: Box, check: Rectangle => Boolean, f: Entry[A] => Boolean): Seq[Entry[A]] = {
    if (!space.isValid) Nil else {
      val buf = ArrayBuffer.empty[Entry[A]]
      def recur(node: Node[A]): Unit = node match {
        case External(children, box) =>
          children.foreach { c =>
            if (check(c.rect) && f(c)) buf.append(c)
          }
        case Internal(children, box) =>
          children.foreach { c =>
            if (space.intersects(box)) recur(c)
          }
      }
      if (space.intersects(box)) recur(this)
      buf
    }
  }

  def foldSearch[B](space: Box, init: B)(f: (B, Entry[A]) => B): B =
    searchIterator(space, _ => true).foldLeft(init)(f)

  def searchIterator(space: Box, f: Entry[A] => Boolean): Iterator[Entry[A]] = {
    if (children.isEmpty || !box.intersects(space)) {
      Iterator.empty
    } else {
      this match {
        case External(cs, _) =>
          cs.iterator.filter(c => space.contains(c.rect) && f(c))
        case Internal(cs, _) =>
          cs.iterator.flatMap(c => c.searchIterator(space, f))
      }
    }
  }

  def nearest(pt: Point, d0: Double): Option[(Double, Entry[A])] = {
    var dist: Double = d0
    var result: Option[(Double, Entry[A])] = None
    this match {
      case External(children, box) =>
        children.foreach { entry =>
          val d = entry.rect.distanceActual(pt)
          if (d < dist) {
            dist = d
            result = Some((d, entry))
          }
        }
      case Internal(children, box) =>
        val cs = children.map(node => (node.box.distanceActual(pt), node)).sortBy(_._1)
        cs.foreach { case (d, node) =>
          if (d >= dist) return result //scalastyle:ignore
          node.nearest(pt, dist) match {
            case some @ Some((d, _)) =>
              dist = d
              result = some
            case None =>
          }
        }
    }
    result
  }

  def nearestK(pt: Point, k: Int, d0: Double, pq: PriorityQueue[(Double, Entry[A])]): Double = {
    var dist: Double = d0
    this match {
      case External(children, box) =>
        children.foreach { entry =>
          val d = entry.rect.distanceActual(pt)
          if (d < dist) {
            pq += ((d, entry))
            if (pq.size > k) {
              pq.dequeue
              dist = pq.head._1
            }
          }
        }
      case Internal(children, box) =>
        val cs = children.map(node => (node.box.distanceActual(pt), node)).sortBy(_._1)
        cs.foreach { case (d, node) =>
          if (d >= dist) return dist //scalastyle:ignore
          dist = node.nearestK(pt, k, dist, pq)
        }
    }
    dist
  }

  def count(space: Box): Int = {
    if (!space.isValid) 0 else {
      def recur(node: Node[A]): Int = node match {
        case External(children, box) =>
          var n = 0
          var i = 0
          while (i < children.length) {
            if (space.contains(children(i).rect)) n += 1
            i += 1
          }
          n
        case Internal(children, box) =>
          var n = 0
          var i = 0
          while (i < children.length) {
            val c = children(i)
            if (space.intersects(c.box)) n += recur(c)
            i += 1
          }
          n
      }
      if (space.intersects(box)) recur(this) else 0
    }
  }


  def contains(entry: Entry[A]): Boolean = {
    searchIterator(entry.rect.toBox, _ == entry).nonEmpty
  }

  def map[B](f: A => B): Node[B] = this match {
    case External(cs, box) =>
      External(cs.map(e => Entry(e.rect, f(e.value))), box)
    case Internal(cs, box) =>
      Internal(cs.map(_.map(f)), box)
  }
}

case class Internal[A](children: Vector[Node[A]], box: Box) extends Node[A] {

  def remove(entry: Entry[A]): Option[(ADT[Entry[A]], Option[Node[A]])] = {
    def loop(i: Int): Option[(ADT[Entry[A]], Option[Node[A]])] =
      if (i < children.length) {
        val child = children(i)
        child.remove(entry) match {
          case None =>
            loop(i + 1)

          case Some((es, None)) =>
            if (children.length == 1) {
              Some((es, None))
            } else if (children.length == 2) {
              Some((ADT.wrap(children(1 - i).entries) plus es, None))
            } else {
              val cs = children.take(i) ++ children.drop(i + 1)
              val b = contract(child.rect, cs.foldLeft(Box.empty)(_ expandRectangle _.rect))
              Some((es, Some(Internal(cs, b))))
            }

          case Some((es, Some(node))) =>
            val cs = children.updated(i, node)
            val b = contract(child.rect, cs.foldLeft(Box.empty)(_ expandRectangle _.rect))
            Some((es, Some(Internal(cs, b))))
        }
      } else {
        None
      }

    if (!box.contains(entry.rect)) None else loop(0)
  }
}


case class External[A](children: Vector[Entry[A]], box: Box) extends Node[A] {

  def remove(entry: Entry[A]): Option[(ADT[Entry[A]], Option[Node[A]])] = {
    if (!box.contains(entry.rect)) return None //scalastyle:ignore
    val i = children.indexOf(entry)
    if (i < 0) {
      None
    } else if (children.length == 1) {
      Some((ADT.empty[Entry[A]], None))
    } else if (children.length == 2) {
      Some((ADT(children(1 - i)), None))
    } else {
      val cs = children.take(i) ++ children.drop(i + 1)
      val b = contract(entry.rect, cs.foldLeft(Box.empty)(_ expandRectangle _.rect))
      Some((ADT.empty[Entry[A]], Some(External(cs, b))))
    }
  }
}

case class Entry[A](rect: Rectangle, value: A) extends HasRectangle

object Node {

  def empty[A]: Node[A] = External(Vector.empty, Box.empty)

  def splitExternal[A](children: Vector[Entry[A]]): Vector[External[A]] = {
    val ((es1, box1), (es2, box2)) = splitter(children)
    Vector(External(es1, box1), External(es2, box2))
  }

  def splitInternal[A](children: Vector[Node[A]]): Vector[Internal[A]] = {
    val ((ns1, box1), (ns2, box2)) = splitter(children)
    Vector(Internal(ns1, box1), Internal(ns2, box2))
  }

  def splitter[M <: HasRectangle](children: Vector[M]): ((Vector[M], Box), (Vector[M], Box)) = {
    val buf = ArrayBuffer(children: _*)
    val (seed1, seed2) = pickSeeds(buf)

    var box1: Box = seed1.rect.toBox
    var box2: Box = seed2.rect.toBox
    val nodes1 = ArrayBuffer(seed1)
    val nodes2 = ArrayBuffer(seed2)

    def add1(node: M): Unit = { nodes1 += node; box1 = box1.expandRectangle(node.rect) }
    def add2(node: M): Unit = { nodes2 += node; box2 = box2.expandRectangle(node.rect) }

    while (buf.nonEmpty) {

      if (nodes1.length >= 2 && nodes2.length + buf.length <= 2) {
        // We should put the remaining buffer all in one bucket.
        nodes2 ++= buf
        box2 = buf.foldLeft(box2)(_ expandRectangle _.rect)
        buf.clear()

      } else if (nodes2.length >= 2 && nodes1.length + buf.length <= 2) {
        // We should put the remaining buffer all in the other bucket.
        nodes1 ++= buf
        box1 = buf.foldLeft(box1)(_ expandRectangle _.rect)
        buf.clear()

      } else {
        // We want to find the bucket whose bounding box requires the
        // smallest increase to contain this member. If both are the
        // same, we look for the bucket with the smallest area. If
        // those are the same, we flip a coin.
        val node = buf.remove(buf.length - 1)
        val e1 = box1.expandArea(node.rect)
        val e2 = box2.expandArea(node.rect)
        if (e1 < e2) {
          add1(node)
        } else if (e2 < e1) {
          add2(node)
        } else {
          val b1 = box1.expandRectangle(node.rect)
          val b2 = box2.expandRectangle(node.rect)
          val a1 = b1.area(b1)
          val a2 = b2.area(b2)
          if (a1 < a2) {
            add1(node)
          } else if (a2 < a1) {
            add2(node)
          } else if (Math.random() > 0.5) {
            add1(node)
          } else {
            add2(node)
          }
        }
      }
    }
    ((nodes1.toVector, box1), (nodes2.toVector, box2))
  }

  def pickSeeds[M <: HasRectangle](nodes: ArrayBuffer[M]): (M, M) = {

    // find the two rectangles that have the most space between them
    // in this particular dimension. the sequence is (lower, upper) points
    def handleDimension(pairs: IndexedSeq[(Double, Double)]): (Double, Int, Int) = {

      val (a0, b0) = pairs(0)
      var amin = a0 // min lower coord
      var amax = a0 // max lower coord
      var bmin = b0 // min upper coord
      var bmax = b0 // max upper coord

      var left = 0
      var right = 0
      var i = 1
      while (i < pairs.length) {
        val (a, b) = pairs(i)
        if (a < amin) { amin = a }
        if (a > amax) { amax = a; right = i }
        if (b > bmax) { bmax = b }
        if (b < bmin) { bmin = b; left = i }
        i += 1
      }

      if (left != right) ((bmin - amax) / (bmax - amin), left, right) else (0.0, 0, 1)
    }
    val dimension = nodes.toList(0).rect.dimension
    // get back the maximum distance in each dimension, and the coords
    var tupleList: List[(Double, Int, Int)] = List.fill(nodes.toList(0).rect.dimension)((0, 0, 0))

    for(i <- 0 until dimension) {
      tupleList = tupleList.updated(i, handleDimension(nodes.map(n => (n.rect.bottomLeft(i), n.rect.topRight(i)))))
    }

    var (max, i, j) = tupleList(0)
    for(iter <- 1 until dimension) {
      if(max < tupleList(iter)._1) {
        max = tupleList(iter)._1
        i = tupleList(iter)._2
        j = tupleList(iter)._3
      }
    }

//    val (i, j) = if (w1 > w2) (i1, j1) else (i2, j2)

    // remove these nodes and return them
    // make sure to remove the larger index first.
    val (a, b) = if (i > j) (i, j) else (j, i)
    val node1 = nodes.remove(a)
    val node2 = nodes.remove(b)
    (node1, node2)
  }
}
