package org.apache.mahout.clustering;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.TaskAttemptID;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.common.DummyOutputCollector;

public class MockReducerContext<K extends WritableComparable, V extends Writable> extends Context {

  private DummyOutputCollector<K, V> collector;

  public MockReducerContext(Reducer<?,?,?,?> reducer, Configuration conf, DummyOutputCollector<K, V> collector, Class keyIn,
      Class<?> valueIn) throws IOException, InterruptedException {
    reducer.super(conf, new TaskAttemptID(), new MockIterator(), null, null, null, null, null, null, keyIn, valueIn);
    this.collector = collector;
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.TaskInputOutputContext#setStatus(java.lang.String)
   */
  @Override
  public void setStatus(String status) {
    // TODO Auto-generated method stub
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.TaskInputOutputContext#write(java.lang.Object, java.lang.Object)
   */
  @Override
  public void write(Object key, Object value) throws IOException, InterruptedException {
    collector.collect((K) key, (V) value);
  }

}
