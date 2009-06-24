package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.VectorIterable;
import org.apache.mahout.matrix.Vector;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;

import java.io.IOException;


/**
 * Closes the writer when done
 *
 **/
public class SequenceFileVectorWriter implements VectorWriter {
  protected SequenceFile.Writer writer;

  public SequenceFileVectorWriter(SequenceFile.Writer writer) {
    this.writer = writer;
  }

  @Override
  public long write(VectorIterable iterable, long maxDocs) throws IOException {
    long recNum = 0;
    for (Vector point : iterable) {
      if (recNum >= maxDocs) {
        break;
      }
      //point.write(dataOut);
      writer.append(new LongWritable(recNum++), point);

    }
    return recNum;
  }

  @Override
  public long write(VectorIterable iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }

  @Override
  public void close() throws IOException {
    if (writer != null) {
      writer.close();
    }
  }

  public SequenceFile.Writer getWriter() {
    return writer;
  }
}
