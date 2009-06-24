package org.apache.mahout.utils.vectors;

import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.Vector;

import java.util.Iterator;
import java.io.IOException;


/**
 * Reads in a file containing {@link org.apache.mahout.matrix.Vector}s and provides
 * a {@link org.apache.mahout.utils.vectors.VectorIterable} interface to them.
 * <p/>
 * The key is any {@link org.apache.hadoop.io.Writable} and the value is a {@link org.apache.mahout.matrix.Vector}.
 * It can handle any class that implements Vector as long as it has a no-arg constructor.
 *
 **/
public class SequenceFileVectorIterable implements VectorIterable {
  private SequenceFile.Reader reader;

  public SequenceFileVectorIterable(SequenceFile.Reader reader) {
    this.reader = reader;
  }


  @Override
  public Iterator<Vector> iterator() {
    try {
      return new SeqFileIterator();
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
  }

  private class SeqFileIterator implements Iterator<Vector> {
    private Writable key;
    private Vector value;

    private SeqFileIterator() throws IllegalAccessException, InstantiationException {
      value = (Vector) reader.getValueClass().newInstance();
      key = (Writable) reader.getKeyClass().newInstance();
    }

    @Override
    public boolean hasNext() {
      try {
        return reader.next(key, value);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public Vector next() {
      return value;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
