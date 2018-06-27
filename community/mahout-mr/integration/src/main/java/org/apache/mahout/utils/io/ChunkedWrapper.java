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

package org.apache.mahout.utils.io;

import java.io.IOException;

/**
 * {@link ChunkedWriter} based implementation of the {@link WrappedWriter} interface.
 */
public class ChunkedWrapper implements WrappedWriter {

  private final ChunkedWriter writer;

  public ChunkedWrapper(ChunkedWriter writer) {
    this.writer = writer;
  }

  @Override
  public void write(String key, String value) throws IOException {
    writer.write(key, value);
  }

  @Override
  public void close() throws IOException {
    writer.close();
  }
}
