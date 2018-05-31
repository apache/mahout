/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings.drm;

import water.fvec.Frame;
import water.fvec.Vec;

/**
 * Class which represents a Mahout DRM in H2O.
 */
public class H2ODrm {
  /** frame stores all the numerical data of a DRM. */
  public Frame frame;
  /** keys stores the row key bindings (String or Long) */
  public Vec keys;

  /**
   * Class constructor. Null key represents Int keyed DRM.
   */
  public H2ODrm(Frame m) {
    frame = m;
    keys = null;
  }

  /**
   * Class constructor. Both Numerical and row key bindings specified.
   */
  public H2ODrm(Frame m, Vec k) {
    frame = m;
    keys = k;
  }
}
