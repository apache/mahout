/**
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

package org.apache.mahout.math.hadoop.stochasticsvd.qr;

import java.io.Closeable;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.Validate;
import org.apache.mahout.common.iterator.CopyConstructorIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.DenseBlockWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.UpperTriangular;

import com.google.common.collect.Lists;

/**
 * Second/last step of QR iterations. Takes input of qtHats and rHats and
 * provides iterator to pull ready rows of final Q.
 * 
 */
public class QRLastStep implements Closeable, Iterator<Vector> {

  private Iterator<DenseBlockWritable> qHatInput;

  private final List<UpperTriangular> mRs = Lists.newArrayList();
  private final int blockNum;
  private double[][] mQt;
  private int cnt;
  private int r;
  private int kp;
  private Vector qRow;

  /**
   * 
   * @param qHatInput
   *          the Q-Hat input that was output in the first step
   * @param rHatInput
   *          all RHat outputs int the group in order of groups
   * @param blockNum
   *          our RHat number in the group
   * @throws IOException
   */
  public QRLastStep(Iterator<DenseBlockWritable> qHatInput,
                    Iterator<VectorWritable> rHatInput,
                    int blockNum) throws IOException {
    super();
    this.blockNum = blockNum;
    this.qHatInput = qHatInput;
    // in this implementation we actually preload all Rs into memory to make R
    // sequence modifications more efficient.
    int block = 0;
    while (rHatInput.hasNext()) {
      Vector value = rHatInput.next().get();
      if (block < blockNum && block > 0) {
        GivensThinSolver.mergeR(mRs.get(0), new UpperTriangular(value));
      } else {
        mRs.add(new UpperTriangular(value));
      }
      block++;
    }

  }

  private boolean loadNextQt() {
    DenseBlockWritable v = new DenseBlockWritable();

    boolean more = qHatInput.hasNext();
    if (!more)
      return false;

    v = qHatInput.next();

    mQt =
      GivensThinSolver
        .computeQtHat(v.getBlock(),
                      blockNum == 0 ? 0 : 1,
                      new CopyConstructorIterator<UpperTriangular>(mRs
                        .iterator()));
    r = mQt[0].length;
    kp = mQt.length;
    if (qRow == null) {
      qRow = new DenseVector(kp);
    }
    return true;
  }

  @Override
  public boolean hasNext() {
    boolean result = true;
    if (mQt != null && cnt == r) {
      mQt = null;
    }
    if (mQt == null) {
      result = loadNextQt();
      cnt = 0;
    }
    return result;
  }

  @Override
  public Vector next() {
    Validate.isTrue(hasNext(), "Q input overrun");
    int qRowIndex = r - cnt - 1; // because QHats are initially stored in
    for (int j = 0; j < kp; j++)
      qRow.setQuick(j, mQt[j][qRowIndex]);
    cnt++;
    return qRow;
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void close() throws IOException {
    mQt = null;

    mRs.clear();
  }

}
