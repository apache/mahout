package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.VectorIterable;

import java.io.IOException;
import java.io.Writer;


/**
 *
 *
 **/
public class JWriterVectorWriter implements VectorWriter {
  protected Writer writer;

  public JWriterVectorWriter(Writer writer) {
    this.writer = writer;
  }

  @Override
  public long write(VectorIterable iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }

  @Override
  public long write(VectorIterable iterable, long maxDocs) throws IOException {
    long result = 0;

    for (Vector vector : iterable) {
      if (result >= maxDocs) {
        break;
      }
      writer.write(vector.asFormatString());
      writer.write("\n");

      result++;
    }
    return result;
  }

  @Override
  public void close() throws IOException {

  }
}
