package org.apache.mahout.utils.vectors.arff;

public enum ARFFType {
  NUMERIC("numeric"), NOMINAL("{"), DATE("date"), STRING("string");

  private String indicator;
  ARFFType(String indicator) {
    this.indicator = indicator;
  }

  public String getIndicator() {
    return indicator;
  }

  

  public String getLabel(String line) {
    int idx = line.indexOf(indicator);
    return line.substring(ARFFModel.ATTRIBUTE.length(),
            idx).trim();
  }
}
