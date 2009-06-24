package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.VectorIterable;

import java.io.IOException;


/**
 *
 *
 **/
public interface VectorWriter {
  /**
   * Write all values in the Iterable to the output
   * @param iterable The {@link org.apache.mahout.utils.vectors.VectorIterable}
   * @return the number of docs written
   * @throws IOException if there was a problem writing
   *
   * @see #write(org.apache.mahout.utils.vectors.VectorIterable, long)
   */
  public long write(VectorIterable iterable) throws IOException;

  /**
   * Write the first <code>maxDocs</code> to the output.
   * @param iterable The {@link org.apache.mahout.utils.vectors.VectorIterable}
   * @param maxDocs the maximum number of docs to write
   * @return The number of docs written
   * @throws IOException if there was a problem writing
   */
  public long write(VectorIterable iterable, long maxDocs) throws IOException;

  /**
   * Close any internally held resources.  If external Writers are passed in, the implementation should indicate
   * whether it also closes them
   * @throws IOException if there was an issue closing the item
   */
  public void close() throws IOException;
}
