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
import scala.math.{min, max}
import scala.util.Try

object RTree {

  def empty[A]: RTree[A] = new RTree(Node.empty[A], 0)
  def apply[A](entries: Entry[A]*): RTree[A] =
    entries.foldLeft(RTree.empty[A])(_ insert _)
}

case class RTree[A](root: Node[A], size: Int) {

  def ===(that: RTree[A]): Boolean =
    size == that.size && entries.forall(that.contains)


  override def equals(that: Any): Boolean =
    that match {
      case rt: RTree[_] =>
        Try(this === rt.asInstanceOf[RTree[A]]).getOrElse(false)
      case _ =>
        false
    }

  override def hashCode(): Int = {
    var x = 0xbadd0995
    val it = entries
    while (it.hasNext) x ^= (it.next.hashCode * 777 + 1)
    x
  }

  def insert(coordinates: List[Double], value: A): RTree[A] =
    insert(Entry(Point(coordinates), value))

  def insert(entry: Entry[A]): RTree[A] = {
    val r = root.insert(entry) match {
      case Left(rs) => Internal(rs, rs.foldLeft(Box.empty)(_ expandRectangle _.box))
      case Right(r) => r
    }
    RTree(r, size + 1)
  }

  def insertAll(entries: Iterable[Entry[A]]): RTree[A] =
    entries.foldLeft(this)(_ insert _)

  def remove(entry: Entry[A]): RTree[A] =
    root.remove(entry) match {
      case None =>
        this
      case Some((es, None)) =>
        es.foldLeft(RTree.empty[A])(_ insert _)
      case Some((es, Some(node))) =>
        es.foldLeft(RTree(node, size - es.size - 1))(_ insert _)
    }

  def removeAll(entries: Iterable[Entry[A]]): RTree[A] =
    entries.foldLeft(this)(_ remove _)

  def search(space: Box): Seq[Entry[A]] =
    root.search(space, _ => true)

  def search(space: Box, f: Entry[A] => Boolean): Seq[Entry[A]] =
    root.search(space, f)

  def searchIntersection(space: Box): Seq[Entry[A]] =
    root.searchIntersection(space, _ => true)

  def searchIntersection(space: Box, f: Entry[A] => Boolean): Seq[Entry[A]] =
    root.searchIntersection(space, f)


  def foldSearch[B](space: Box, init: B)(f: (B, Entry[A]) => B): B =
    root.foldSearch(space, init)(f)

  def nearest(pt: Point): Option[Entry[A]] =
    root.nearest(pt, Float.PositiveInfinity).map(_._2)

  def nearestK(pt: Point, k: Int): IndexedSeq[Entry[A]] =
    if (k < 1) {
      Vector.empty
    } else {
      implicit val ord = Ordering.by[(Double, Entry[A]), Double](_._1)
      val pq = PriorityQueue.empty[(Double, Entry[A])]
      root.nearestK(pt, k, Double.PositiveInfinity, pq)
      val arr = new Array[Entry[A]](pq.size)
      var i = arr.length - 1
      while (i >= 0) {
        val (_, e) = pq.dequeue
        arr(i) = e
        i -= 1
      }
      arr
    }

  def count(space: Box): Int =
    root.count(space)

  def contains(coordinates: List[Double], value: A): Boolean =
    root.contains(Entry(Point(coordinates), value))

  def contains(entry: Entry[A]): Boolean =
    root.contains(entry)

  def map[B](f: A => B): RTree[B] =
    RTree(root.map(f), size)

  def entries: Iterator[Entry[A]] =
    root.iterator

  def values: Iterator[A] =
    entries.map(_.value)

//  def pretty: String = root.pretty

  override def toString: String =
    s"RTree(<$size entries>)"
}

