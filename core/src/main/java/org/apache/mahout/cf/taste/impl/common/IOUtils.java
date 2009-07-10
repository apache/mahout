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

package org.apache.mahout.cf.taste.impl.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.charset.Charset;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/** <p>I/O-related utility methods that don't have a better home.</p> */
public final class IOUtils {

  private static final Logger log = LoggerFactory.getLogger(IOUtils.class);

  private IOUtils() {
  }

  public static void quietClose(Closeable closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (IOException ioe) {
        log.warn("Unexpected exception while closing " + closeable + "; continuing", ioe);
      }
    }
  }

  // Sheez, why can't ResultSet, Statement and Connection implement Closeable?

  public static void quietClose(ResultSet closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing " + closeable + "; continuing", sqle);
      }
    }
  }

  public static void quietClose(Statement closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing " + closeable + "; continuing", sqle);
      }
    }
  }

  public static void quietClose(Connection closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing " + closeable + "; continuing", sqle);
      }
    }
  }

  /**
   * Closes a {@link ResultSet}, {@link Statement} and {@link Connection} (if not null) and logs (but does not rethrow)
   * any resulting {@link SQLException}. This is useful for cleaning up after a database query.
   *
   * @param resultSet  {@link ResultSet} to close
   * @param statement  {@link Statement} to close
   * @param connection {@link Connection} to close
   */
  public static void quietClose(ResultSet resultSet, Statement statement, Connection connection) {
    quietClose(resultSet);
    quietClose(statement);
    quietClose(connection);
  }

  public static void copyStreamToFile(InputStream in, File file) throws IOException {
    OutputStream out = new FileOutputStream(file, false);
    copyInToOut(in, out);
  }

  public static String readURL(URL url) throws IOException {
    InputStream in = url.openStream();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    copyInToOut(in, out);
    return new String(out.toByteArray(), Charset.forName("UTF-8"));
  }

  public static File createTempFile(String copySuffixFrom) throws IOException {
    int period = copySuffixFrom.lastIndexOf('.');
    String suffix = period < 0 ? "" : copySuffixFrom.substring(period);
    File tempFile = File.createTempFile("mahout-taste-", suffix);
    tempFile.deleteOnExit();
    return tempFile;
  }

  public static void copyInToOut(InputStream in, OutputStream out) throws IOException {
    try {
      int bytesRead;
      byte[] buffer = new byte[65536];
      while ((bytesRead = in.read(buffer)) > 0) {
        out.write(buffer, 0, bytesRead);
      }
    } finally {
      quietClose(in);
      quietClose(out);
    }
  }

}
