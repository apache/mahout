package org.apache.mahout.utils.parameters;

import org.apache.hadoop.mapred.JobConf;

public class StringParameter extends AbstractParameter<String> {


  public StringParameter(String prefix, String name, JobConf jobConf, String defaultValue, String description) {
    super(String.class, prefix, name, jobConf, defaultValue, description);
  }

  public void setStringValue(String stringValue) {
    set(stringValue);
  }

  public String getStringValue() {
    return get();
  }
}
