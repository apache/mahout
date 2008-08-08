package org.apache.mahout.utils.parameters;

import org.apache.hadoop.mapred.JobConf;

public class IntegerParameter extends AbstractParameter<Integer> {

  public IntegerParameter(String prefix, String name, JobConf jobConf, Integer defaultValue, String description) {
    super(Integer.class, prefix, name, jobConf, defaultValue, description);
  }

  public void setStringValue(String stringValue) {
    set(Integer.valueOf(stringValue));
  }

}
