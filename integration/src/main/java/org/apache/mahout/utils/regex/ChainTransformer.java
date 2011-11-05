package org.apache.mahout.utils.regex;


import java.util.ArrayList;
import java.util.List;

/**
 * Chain together several {@link org.apache.mahout.utils.regex.RegexTransformer} and apply them to the match
 * in succession
 *
 **/
public class ChainTransformer implements RegexTransformer {

  private List<RegexTransformer> chain = new ArrayList<RegexTransformer>();

  public ChainTransformer() {
  }

  public ChainTransformer(List<RegexTransformer> chain) {
    this.chain = chain;
  }

  @Override
  public String transformMatch(String match) {
    String result = match;
    for (RegexTransformer transformer : chain) {
      result = transformer.transformMatch(result);
    }
    return result;
  }

  public List<RegexTransformer> getChain() {
    return chain;
  }

  public void setChain(List<RegexTransformer> chain) {
    this.chain = chain;
  }
}
