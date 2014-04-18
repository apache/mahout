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

package org.apache.mahout.common;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Collection;

import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * I/O-related utility methods that don't have a better home.
 * </p>
 */
public final class IOUtils {
  
  private static final Logger log = LoggerFactory.getLogger(IOUtils.class);
  
  private IOUtils() { }

  // Sheez, why can't ResultSet, Statement and Connection implement Closeable?
  
  public static void quietClose(ResultSet closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing; continuing", sqle);
      }
    }
  }
  
  public static void quietClose(Statement closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing; continuing", sqle);
      }
    }
  }
  
  public static void quietClose(Connection closeable) {
    if (closeable != null) {
      try {
        closeable.close();
      } catch (SQLException sqle) {
        log.warn("Unexpected exception while closing; continuing", sqle);
      }
    }
  }
  
  /**
   * Closes a {@link ResultSet}, {@link Statement} and {@link Connection} (if not null) and logs (but does not
   * rethrow) any resulting {@link SQLException}. This is useful for cleaning up after a database query.
   * 
   * @param resultSet
   *          {@link ResultSet} to close
   * @param statement
   *          {@link Statement} to close
   * @param connection
   *          {@link Connection} to close
   */
  public static void quietClose(ResultSet resultSet, Statement statement, Connection connection) {
    quietClose(resultSet);
    quietClose(statement);
    quietClose(connection);
  }
  
  /**
   * make sure to close all sources, log all of the problems occurred, clear
   * {@code closeables} (to prevent repeating close attempts), re-throw the
   * last one at the end. Helps resource scope management (e.g. compositions of
   * {@link Closeable}s objects)
   * <P>
   * <p/>
   * Typical pattern:
   * <p/>
   *
   * <pre>
   *   LinkedList<Closeable> closeables = new LinkedList<Closeable>();
   *   try {
   *      InputStream stream1 = new FileInputStream(...);
   *      closeables.addFirst(stream1);
   *      ...
   *      InputStream streamN = new FileInputStream(...);
   *      closeables.addFirst(streamN);
   *      ...
   *   } finally {
   *      IOUtils.close(closeables);
   *   }
   * </pre>
   *
   * @param closeables
   *          must be a modifiable collection of {@link Closeable}s
   * @throws IOException
   *           the last exception (if any) of all closed resources
   */
  public static void close(Collection<? extends Closeable> closeables)
    throws IOException {
    Throwable lastThr = null;

    for (Closeable closeable : closeables) {
      try {
        closeable.close();
      } catch (Throwable thr) {
        log.error(thr.getMessage(), thr);
        lastThr = thr;
      }
    }

    // make sure we don't double-close
    // but that has to be modifiable collection
    closeables.clear();

    if (lastThr != null) {
      if (lastThr instanceof IOException) {
        throw (IOException) lastThr;
      } else if (lastThr instanceof RuntimeException) {
        throw (RuntimeException) lastThr;
      } else if (lastThr instanceof Error) {
        throw (Error) lastThr;
      } else {
        // should not happen
        throw (IOException) new IOException("Unexpected exception during close")
            .initCause(lastThr);
      }
    }

  }


  /**
   * for temporary files, a file may be considered as a {@link Closeable} too,
   * where file is wiped on close and thus the disk resource is released
   * ('closed').
   * 
   * 
   */
  public static class DeleteFileOnClose implements Closeable {

    private final File file;

    public DeleteFileOnClose(File file) {
      this.file = file;
    }

    @Override
    public void close() throws IOException {
      if (file.isFile()) {
        file.delete();
      }
    }
  }

  /**
   * MultipleOutputs to closeable adapter.
   * 
   */
  public static class MultipleOutputsCloseableAdapter implements Closeable {
    private final MultipleOutputs mo;

    public MultipleOutputsCloseableAdapter(MultipleOutputs mo) {
      this.mo = mo;
    }

    @Override
    public void close() throws IOException {
      if (mo != null) {
        mo.close();
      }
    }
  }

}
