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

package org.apache.mahout

import org.apache.log4j.{Level, Priority, Logger}

package object logging {

  /** Compute `expr` if debug is on, only */
  def debugDo[T](expr: => T)(implicit log: Logger): Option[T] = {
    if (log.isDebugEnabled) Some(expr)
    else None
  }

  /** Compute `expr` if trace is on, only */
  def traceDo[T](expr: => T)(implicit log: Logger): Option[T] = {
    if (log.isTraceEnabled) Some(expr) else None
  }

  /** Shorter, and lazy, versions of logging methods. Just declare log implicit. */
  def debug(msg: => AnyRef)(implicit log: Logger) { if (log.isDebugEnabled) log.debug(msg) }

  def debug(msg: => AnyRef, t: Throwable)(implicit log: Logger) { if (log.isDebugEnabled()) log.debug(msg, t) }

  /** Shorter, and lazy, versions of logging methods. Just declare log implicit. */
  def trace(msg: => AnyRef)(implicit log: Logger) { if (log.isTraceEnabled) log.trace(msg) }

  def trace(msg: => AnyRef, t: Throwable)(implicit log: Logger) { if (log.isTraceEnabled()) log.trace(msg, t) }

  def info(msg: => AnyRef)(implicit log: Logger) { if (log.isInfoEnabled) log.info(msg)}

  def info(msg: => AnyRef, t:Throwable)(implicit log: Logger) { if (log.isInfoEnabled) log.info(msg,t)}

  def warn(msg: => AnyRef)(implicit log: Logger) { if (log.isEnabledFor(Level.WARN)) log.warn(msg) }

  def warn(msg: => AnyRef, t: Throwable)(implicit log: Logger) { if (log.isEnabledFor(Level.WARN)) error(msg, t) }

  def error(msg: => AnyRef)(implicit log: Logger) { if (log.isEnabledFor(Level.ERROR)) log.warn(msg) }

  def error(msg: => AnyRef, t: Throwable)(implicit log: Logger) { if (log.isEnabledFor(Level.ERROR)) error(msg, t) }

  def fatal(msg: => AnyRef)(implicit log: Logger) { if (log.isEnabledFor(Level.FATAL)) log.fatal(msg) }

  def fatal(msg: => AnyRef, t: Throwable)(implicit log: Logger) { if (log.isEnabledFor(Level.FATAL)) log.fatal(msg, t) }

  def getLog(name: String): Logger = Logger.getLogger(name)

  def getLog(clazz: Class[_]): Logger = Logger.getLogger(clazz)

  def mahoutLog :Logger = getLog("org.apache.mahout")

  def setLogLevel(l:Level)(implicit log:Logger) = {
    log.setLevel(l)
  }

  def setAdditivity(a:Boolean)(implicit log:Logger) = log.setAdditivity(a)

}
