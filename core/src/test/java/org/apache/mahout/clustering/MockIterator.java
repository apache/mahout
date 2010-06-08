package org.apache.mahout.clustering;

import java.io.IOException;

import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.mapred.RawKeyValueIterator;
import org.apache.hadoop.util.Progress;

public class MockIterator implements RawKeyValueIterator {

  @Override
  public void close() throws IOException {
  }

  @Override
  public DataInputBuffer getKey() throws IOException {
    return null;
  }

  @Override
  public Progress getProgress() {
    return null;
  }

  @Override
  public DataInputBuffer getValue() throws IOException {

    return null;
  }

  @Override
  public boolean next() throws IOException {
    return true;
  }

}
