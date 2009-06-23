package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.VectorIterable;

import java.io.IOException;


/**
 *
 *
 **/
public interface VectorWriter {
  public long write(VectorIterable iterable) throws IOException;

  public long write(VectorIterable iterable, long maxDocs) throws IOException;

  public void close() throws IOException;
}
