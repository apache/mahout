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

import java.io.BufferedReader;
import java.io.IOException;

/**
 * Uses the same logic as TrainLogistic and RunLogistic for finding an input, but instead
 * of processing the input, this class just prints the input to standard out.
 */
public class PrintResourceOrFile {
  public static void main(String[] args) throws IOException {
    if (args.length != 1) {
      throw new IllegalArgumentException("Must have a single argument that names a file or resource.");
    }
    BufferedReader in = TrainLogistic.InputOpener.open(args[0]);
    String line = in.readLine();
    while (line != null) {
      System.out.println(line);
      line = in.readLine();
    }
  }
}
