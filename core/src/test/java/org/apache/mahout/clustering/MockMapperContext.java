package org.apache.mahout.clustering;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.common.DummyOutputCollector;

public class MockMapperContext<K extends WritableComparable, V extends Writable> extends Context {

  private final DummyOutputCollector<K, V> collector;

  public MockMapperContext(Mapper<?,?,?,?> mapper, Configuration arg0,
      DummyOutputCollector<K,V> collector) throws IOException, InterruptedException {
    mapper.super(arg0, new TaskAttemptID(), null, null, null, null, null);
    this.collector = collector;
  }

  @Override
  public void write(Object key, Object value) throws IOException, InterruptedException {
    collector.collect((K)key, (V)value);
  }

  @Override
  public void setStatus(String status) {
    // TODO Auto-generated method stub
  }

}
