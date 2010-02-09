package org.apache.mahout.math.matrix.impl;


/**
 * Factory producing implementations of {@link org.apache.mahout.math.matrix.impl.Former} via method create();
 * Implementations of can use existing libraries such as PrintfFormat or Format or other.
 * Serves to isolate the interface of String formatting from the actual implementation.
 * If you want to plug in a different String formatting implementation, simply replace this class with your alternative.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class FormerFactory {

  /**
   * Constructs and returns a new format instance.
   *
   * @param format the format string following printf conventions. The string has a prefix, a format code and a suffix.
   *               The prefix and suffix become part of the formatted output. The format code directs the formatting of
   *               the (single) parameter to be formatted. The code has the following structure <ul> <li> a % (required)
   *               <li> a modifier (optional) <dl> <dt> + <dd> forces display of + for positive numbers <dt> 0 <dd> show
   *               leading zeroes <dt> - <dd> align left in the field <dt> space <dd> prepend a space in front of
   *               positive numbers <dt> # <dd> use "alternate" format. Add 0 or 0x for octal or hexadecimal numbers.
   *               Don't suppress trailing zeroes in general floating point format. </dl> <li> an integer denoting field
   *               width (optional) <li> a period followed by an integer denoting precision (optional) <li> a format
   *               descriptor (required) <dl> <dt>f <dd> floating point number in fixed format <dt>e, E <dd> floating
   *               point number in exponential notation (scientific format). The E format results in an uppercase E for
   *               the exponent (1.14130E+003), the e format in a lowercase e. <dt>g, G <dd> floating point number in
   *               general format (fixed format for small numbers, exponential format for large numbers). Trailing
   *               zeroes are suppressed. The G format results in an uppercase E for the exponent (if any), the g format
   *               in a lowercase e. <dt>d, i <dd> integer in decimal <dt>x <dd> integer in hexadecimal <dt>o <dd>
   *               integer in octal <dt>s <dd> string <dt>c <dd> character </dl> </ul>
   * @throws IllegalArgumentException if bad format
   */
  public Former create(String format) {
    return new Former() {
      public String form(double value) {
        return String.valueOf(value);
      }
    };
  }
}
