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

package org.apache.mahout.classifier.sequencelearning.hmm;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Utils for serializing Writable parts of HmmModel (that means without hidden state names and so on)
 */
final class LossyHmmSerializer {

  private LossyHmmSerializer() {
  }

  static void serialize(HmmModel model, DataOutput output) throws IOException {
    MatrixWritable matrix = new MatrixWritable(model.getEmissionMatrix());
    matrix.write(output);
    matrix.set(model.getTransitionMatrix());
    matrix.write(output);

    VectorWritable vector = new VectorWritable(model.getInitialProbabilities());
    vector.write(output);
  }

  static HmmModel deserialize(DataInput input) throws IOException {
    MatrixWritable matrix = new MatrixWritable();
    matrix.readFields(input);
    Matrix emissionMatrix = matrix.get();

    matrix.readFields(input);
    Matrix transitionMatrix = matrix.get();

    VectorWritable vector = new VectorWritable();
    vector.readFields(input);
    Vector initialProbabilities = vector.get();

    return new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);
  }

}
