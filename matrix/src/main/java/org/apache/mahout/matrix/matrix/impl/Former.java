/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.impl;

/**
 * Formats a double into a string (like sprintf in C).
 *
 * @see java.util.Comparator
 * @see org.apache.mahout.colt
 * @see org.apache.mahout.matrix.Sorting
 */
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public interface Former {
/** 
 * Formats a double into a string (like sprintf in C).
 * @param x the number to format
 * @return the formatted string 
 * @exception IllegalArgumentException if bad argument
 */
String form(double value);
}
