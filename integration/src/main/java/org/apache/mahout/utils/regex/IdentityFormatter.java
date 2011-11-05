package org.apache.mahout.utils.regex;


/**
 *
 *
 **/
public class IdentityFormatter implements RegexFormatter {

  @Override
  public String format(String toFormat) {
    return toFormat;
  }
}
