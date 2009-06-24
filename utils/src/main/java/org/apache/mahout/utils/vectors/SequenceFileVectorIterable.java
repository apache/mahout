package org.apache.mahout.utils.vectors;

import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.Vector;

import java.util.Iterator;
import java.io.IOException;


/**
 *
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
