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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;

import java.io.BufferedReader;

/**
 * Uses the same logic as TrainLogistic and RunLogistic for finding an input, but instead
 * of processing the input, this class just prints the input to standard out.
 */
public final class PrintResourceOrFile {

  private PrintResourceOrFile() {
  }

  public static void main(String[] args) throws Exception {
    Preconditions.checkArgument(args.length == 1, "Must have a single argument that names a file or resource.");
    BufferedReader in = TrainLogistic.open(args[0]);
    try {
      String line;
      while ((line = in.readLine()) != null) {
        System.out.println(line);
      }
    } finally {
      Closeables.close(in, true);
    }
  }
}
