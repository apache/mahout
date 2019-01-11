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

package org.apache.mahout.util

import org.apache.mahout.logging._
import collection._
import java.io.Closeable

object IOUtilsScala {

  private final implicit val log = getLog(IOUtilsScala.getClass)

  /**
   * Try to close every resource in the sequence, in order of the sequence.
   *
   * Report all encountered exceptions to logging.
   *
   * Rethrow last exception only (if any)
   * @param closeables
   */
  def close(closeables: Seq[Closeable]) = {

    var lastThr: Option[Throwable] = None
    closeables.foreach { c =>
      try {
        c.close()
      } catch {
        case t: Throwable =>
          error(t.getMessage, t)
          lastThr = Some(t)
      }
    }

    // Rethrow most recent close exception (can throw only one)
    lastThr.foreach(throw _)
  }

  /**
   * Same as [[IOUtilsScala.close( )]] but do not re-throw any exceptions.
   * @param closeables
   */
  def closeQuietly(closeables: Seq[Closeable]) = {
    try {
      close(closeables)
    } catch {
      case t: Throwable => // NOP
    }
  }
}
