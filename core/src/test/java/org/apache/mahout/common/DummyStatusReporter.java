package org.apache.mahout.common;

import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.StatusReporter;

public class DummyStatusReporter extends StatusReporter {

  Map<Enum<?>, Counter> counters = new HashMap<Enum<?>, Counter>();

  @Override
  public Counter getCounter(Enum<?> name) {
    if (!counters.containsKey(name))
      counters.put(name, new DummyCounter());
    return counters.get(name);
  }

  Map<String, Counter> counterGroups = new HashMap<String, Counter>();

  @Override
  public Counter getCounter(String group, String name) {
    if (!counterGroups.containsKey(group + name))
      counterGroups.put(group + name, new DummyCounter());
    return counterGroups.get(group+name);
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
