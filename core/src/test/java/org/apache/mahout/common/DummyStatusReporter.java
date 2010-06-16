package org.apache.mahout.common;

import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.StatusReporter;

public class DummyStatusReporter extends StatusReporter {

  @Override
  public Counter getCounter(Enum<?> name) {
    throw new NotImplementedException();
  }

  @Override
  public Counter getCounter(String group, String name) {
    throw new NotImplementedException();
  }

  @Override
  public void progress() {
    // TODO Auto-generated method stub   
  }

  @Override
  public void setStatus(String status) {
    // TODO Auto-generated method stub  
  }

}
