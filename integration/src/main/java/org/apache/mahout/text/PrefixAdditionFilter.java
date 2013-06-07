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

package org.apache.mahout.text;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.utils.io.ChunkedWriter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Map;

/**
 * Default parser for parsing text into sequence files.
 */
public final class PrefixAdditionFilter extends SequenceFilesFromDirectoryFilter {

  public PrefixAdditionFilter(Configuration conf,
                              String keyPrefix,
                              Map<String, String> options, 
                              ChunkedWriter writer,
                              Charset charset,
                              FileSystem fs) {
    super(conf, keyPrefix, options, writer, charset, fs);
  }

  @Override
  protected void process(FileStatus fst, Path current) throws IOException {
    FileSystem fs = getFs();
    ChunkedWriter writer = getWriter();
    if (fst.isDir()) {
      String dirPath = getPrefix() + Path.SEPARATOR + current.getName() + Path.SEPARATOR + fst.getPath().getName();
      fs.listStatus(fst.getPath(),
                    new PrefixAdditionFilter(getConf(), dirPath, getOptions(), writer, getCharset(), fs));
    } else {
      InputStream in = null;
      try {
        in = fs.open(fst.getPath());

        StringBuilder file = new StringBuilder();
        for (String aFit : new FileLineIterable(in, getCharset(), false)) {
          file.append(aFit).append('\n');
        }
        String name = current.getName().equals(fst.getPath().getName())
            ? current.getName()
            : current.getName() + Path.SEPARATOR + fst.getPath().getName();
        writer.write(getPrefix() + Path.SEPARATOR + name, file.toString());
      } finally {
        Closeables.close(in, false);
      }
    }
  }
}
