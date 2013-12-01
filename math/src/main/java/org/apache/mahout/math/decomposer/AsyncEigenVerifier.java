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

package org.apache.mahout.math.decomposer;

import java.io.Closeable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

public class AsyncEigenVerifier extends SimpleEigenVerifier implements Closeable {

  private final ExecutorService threadPool;
  private EigenStatus status;
  private boolean finished;
  private boolean started;

  public AsyncEigenVerifier() {
    threadPool = Executors.newFixedThreadPool(1);
    status = new EigenStatus(-1, 0);
  }

  @Override
  public synchronized EigenStatus verify(VectorIterable corpus, Vector vector) {
    if (!finished && !started) { // not yet started or finished, so start!
      status = new EigenStatus(-1, 0);
      Vector vectorCopy = vector.clone();
      threadPool.execute(new VerifierRunnable(corpus, vectorCopy));
      started = true;
    }
    if (finished) {
      finished = false;
    }
    return status;
  }

  @Override
  public void close() {
	  this.threadPool.shutdownNow();
  }
  protected EigenStatus innerVerify(VectorIterable corpus, Vector vector) {
    return super.verify(corpus, vector);
  }

  private class VerifierRunnable implements Runnable {
    private final VectorIterable corpus;
    private final Vector vector;

    protected VerifierRunnable(VectorIterable corpus, Vector vector) {
      this.corpus = corpus;
      this.vector = vector;
    }

    @Override
    public void run() {
      EigenStatus status = innerVerify(corpus, vector);
      synchronized (AsyncEigenVerifier.this) {
        AsyncEigenVerifier.this.status = status;
        finished = true;
        started = false;
      }
    }
  }
}
