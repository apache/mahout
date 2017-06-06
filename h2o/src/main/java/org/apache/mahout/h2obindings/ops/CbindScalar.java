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

package org.apache.mahout.h2obindings.ops;

import org.apache.mahout.h2obindings.drm.H2ODrm;
import water.fvec.Frame;
import water.fvec.Vec;

/**
 * R-like cbind like operator, on a DRM and a new column containing
 * the given scalar value.
 */
public class CbindScalar {
  /**
   * Combine the columns of DRM A with a new column storing
   * the given scalar.
   *
   * @param drmA DRM representing matrix A.
   * @param scalar value to be filled in new column.
   * @param leftbind true if binding to the left
   * @return new DRM containing columns of A and d.
   */
  public static H2ODrm exec(H2ODrm drmA, double scalar, boolean leftbind) {
    Frame fra = drmA.frame;
    Vec newcol = fra.anyVec().makeCon(scalar);
    Vec vecs[] = new Vec[fra.vecs().length + 1];
    int d = 0;

    if (leftbind)
      vecs[d++] = newcol;
    for (Vec vfra : fra.vecs())
      vecs[d++] = vfra;
    if (!leftbind)
      vecs[d++] = newcol;

    return new H2ODrm(new Frame(vecs), drmA.keys);
  }
}
