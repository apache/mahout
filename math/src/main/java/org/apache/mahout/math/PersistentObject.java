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
/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

/**
 * This empty class is the common root for all persistent capable classes.
 * If this class inherits from <tt>java.lang.Object</tt> then all subclasses are serializable with
 * the standard Java serialization mechanism.
 * If this class inherits from <tt>com.objy.db.app.ooObj</tt> then all subclasses are
 * <i>additionally</i> serializable with the Objectivity ODBMS persistance mechanism.
 * Thus, by modifying the inheritance of this class the entire tree of subclasses can
 * be switched to Objectivity compatibility (and back) with minimum effort.
 */
public abstract class PersistentObject implements java.io.Serializable, Cloneable {

  /** Not yet commented. */
  protected PersistentObject() {
  }

  /**
   * Returns a copy of the receiver. This default implementation does not nothing except making the otherwise
   * <tt>protected</tt> clone method <tt>public</tt>.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    try {
      return super.clone();
    } catch (CloneNotSupportedException exc) {
      throw new InternalError(); //should never happen since we are cloneable
    }
  }
}
