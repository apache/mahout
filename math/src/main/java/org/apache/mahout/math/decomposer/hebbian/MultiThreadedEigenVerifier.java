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

package org.apache.mahout.math.decomposer.hebbian;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;


public class MultiThreadedEigenVerifier extends SimpleEigenVerifier {

  private final Executor threadPool;
  private EigenStatus status = null;
  private boolean finished = false;
  private boolean started = false;

  public MultiThreadedEigenVerifier() {
    threadPool = Executors.newFixedThreadPool(1);
    status = new EigenStatus(-1, 0);
  }

  @Override
  public EigenStatus verify(Matrix eigenMatrix, Vector vector) {
    synchronized (status) {
      if (!finished && !started) // not yet started or finished, so start!
      {
        status = new EigenStatus(-1, 0);
        Vector vectorCopy = vector.clone();
        threadPool.execute(new VerifierRunnable(eigenMatrix, vectorCopy));
        started = true;
      }
      if (finished) finished = false;
      return status;
    }
  }

  protected EigenStatus innerVerify(Matrix eigenMatrix, Vector vector) {
    return super.verify(eigenMatrix, vector);
  }

  private class VerifierRunnable implements Runnable {
    private final Matrix eigenMatrix;
    private final Vector vector;

    protected VerifierRunnable(Matrix eigenMatrix, Vector vector) {
      this.eigenMatrix = eigenMatrix;
      this.vector = vector;
    }

    @Override
    public void run() {
      EigenStatus status = innerVerify(eigenMatrix, vector);
      synchronized (MultiThreadedEigenVerifier.this.status) {
        MultiThreadedEigenVerifier.this.status = status;
        finished = true;
        started = false;
      }
    }
  }
}
