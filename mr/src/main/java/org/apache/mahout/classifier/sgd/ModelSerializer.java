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

package org.apache.mahout.classifier.sgd;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import com.google.common.io.Closeables;
import org.apache.hadoop.io.Writable;

/**
 * Provides the ability to store SGD model-related objects as binary files.
 */
public final class ModelSerializer {

  // static class ... don't instantiate
  private ModelSerializer() {
  }

  public static void writeBinary(String path, CrossFoldLearner model) throws IOException {
    DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
    try {
      PolymorphicWritable.write(out, model);
    } finally {
      Closeables.close(out, false);
    }
  }

  public static void writeBinary(String path, OnlineLogisticRegression model) throws IOException {
    DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
    try {
      PolymorphicWritable.write(out, model);
    } finally {
      Closeables.close(out, false);
    }
  }

  public static void writeBinary(String path, AdaptiveLogisticRegression model) throws IOException {
    DataOutputStream out = new DataOutputStream(new FileOutputStream(path));
    try {
      PolymorphicWritable.write(out, model);
    } finally {
      Closeables.close(out, false);
    }
  }

  public static <T extends Writable> T readBinary(InputStream in, Class<T> clazz) throws IOException {
    DataInput dataIn = new DataInputStream(in);
    try {
      return PolymorphicWritable.read(dataIn, clazz);
    } finally {
      Closeables.close(in, false);
    }
  }


}
