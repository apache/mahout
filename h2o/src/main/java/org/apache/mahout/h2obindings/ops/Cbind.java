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

import water.MRTask;
import water.fvec.Frame;
import water.fvec.Vec;
import water.fvec.Chunk;

import org.apache.mahout.h2obindings.drm.H2ODrm;

/**
 * R-like cbind like operator, on two DRMs
 */
public class Cbind {
  /**
   * Combine the columns of two DRMs A and B to create a new DRM.
   *
   * @param drmA DRM representing matrix A.
   * @param drmB DRM representing matrix B.
   * @return new DRM containing columns of A and B adjacent.
   */
  public static H2ODrm exec(H2ODrm drmA, H2ODrm drmB) {
    Frame fra = drmA.frame;
    Vec keysa = drmA.keys;
    Frame frb = drmB.frame;
    Vec keysb = drmB.keys;

    // If A and B are similarly partitioned, ..
    if (fra.anyVec().group() == frb.anyVec().group()) {
      // .. then, do a light weight zip()
      return zip(fra, keysa, frb, keysb);
    } else {
      // .. else, do a heavy weight join() which involves moving data over the wire
      return join(fra, keysa, frb, keysb);
    }
  }

  /** Light weight zip(), no data movement */
  private static H2ODrm zip(final Frame fra, final Vec keysa, final Frame frb, final Vec keysb) {
    // Create a new Vec[] to hold the concatenated list of A and B's column vectors
    Vec vecs[] = new Vec[fra.vecs().length + frb.vecs().length];
    int d = 0;
    // fill A's column vectors
    for (Vec vfra : fra.vecs()) {
      vecs[d++] = vfra;
    }
    // and B's
    for (Vec vfrb : frb.vecs()) {
      vecs[d++] = vfrb;
    }
    // and create a new Frame with the combined list of column Vecs
    Frame fr = new Frame(vecs);
    /* Finally, inherit A's string labels into the result */
    return new H2ODrm(fr, keysa);
  }

  /** Heavy weight join(), involves moving data */
  private static H2ODrm join(final Frame fra, final Vec keysa, final Frame frb, final Vec keysb) {
    // The plan is to re-organize B to be "similarly partitioned as A", and then zip()
    Vec bvecs[] = new Vec[frb.vecs().length];

    for (int i = 0; i < bvecs.length; i++) {
      // First create column Vecs which are similarly partitioned as A
      bvecs[i] = fra.anyVec().makeZero();
    }

    // Next run an MRTask on the new vectors, and fill each cell (initially 0)
    // by pulling in appropriate values from B (frb)
    new MRTask() {
      public void map(Chunk chks[]) {
        int chunkSize = chks[0].len();
        long start = chks[0].start();
        Vec vecs[] = frb.vecs();

        for (int r = 0; r < chunkSize; r++) {
          for (int c = 0; c < chks.length; c++) {
            // assert va.atStr(start+r) == vb.atStr(start+r)
            chks[c].set(r, vecs[c].at(start + r));
          }
        }
      }
    }.doAll(bvecs);

    // now that bvecs[] is compatible, just zip'em'up
    return zip(fra, keysa, new Frame(bvecs), null);
  }
}
