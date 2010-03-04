/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/

package org.apache.mahout.math.jet.random.engine;

/**
 * (Seemingly gigantic) table of good seeds for pseudo-random number generators.
 * <p>
 * <b>Implementation:</b>
 * <dt>This is a port of <tt>SeedTable.h</tt> used in <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep">CLHEP 1.4.0</A> (C++).
 * CLHEP's implementation, in turn, is part of <A HREF="http://wwwinfo.cern.ch/asd/geant/geant4.html">GEANT 4</A>, a C++ simulation toolkit for High Energy Physics.
 * Geant4, in turn,  took the table from the original FORTRAN77 implementation of the HEP CERN Library routine RECUSQ.
 * Each sequence has a period of 10**9 numbers.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class RandomSeedTable {

  /** The number of columns of the matrix (currently COLUMNS = 2). */
  private static final int COLUMNS = 2;

  // a m*n matrix, just stored as one-dim array
  // 215 * 2 entries
  private static final int[] seeds = {
      9876, 54321,
      1299961164, 253987020,
      669708517, 2079157264,
      190904760, 417696270,
      1289741558, 1376336092,
      1803730167, 324952955,
      489854550, 582847132,
      1348037628, 1661577989,
      350557787, 1155446919,
      591502945, 634133404,
      1901084678, 862916278,
      1988640932, 1785523494,
      1873836227, 508007031,
      1146416592, 967585720,
      1837193353, 1522927634,
      38219936, 921609208,
      349152748, 112892610,
      744459040, 1735807920,
      1983990104, 728277902,
      309164507, 2126677523,
      362993787, 1897782044,
      556776976, 462072869,
      1584900822, 2019394912,
      1249892722, 791083656,
      1686600998, 1983731097,
      1127381380, 198976625,
      1999420861, 1810452455,
      1972906041, 664182577,
      84636481, 1291886301,
      1186362995, 954388413,
      2141621785, 61738584,
      1969581251, 1557880415,
      1150606439, 136325185,
      95187861, 1592224108,
      940517655, 1629971798,
      215350428, 922659102,
      786161212, 1121345074,
      1450830056, 1922787776,
      1696578057, 2025150487,
      1803414346, 1851324780,
      1017898585, 1452594263,
      1184497978, 82122239,
      633338765, 1829684974,
      430889421, 230039326,
      492544653, 76320266,
      389386975, 1314148944,
      1720322786, 709120323,
      1868768216, 1992898523,
      443210610, 811117710,
      1191938868, 1548484733,
      616890172, 159787986,
      935835339, 1231440405,
      1058009367, 1527613300,
      1463148129, 1970575097,
      1795336935, 434768675,
      274019517, 605098487,
      483689317, 217146977,
      2070804364, 340596558,
      930226308, 1602100969,
      989324440, 801809442,
      410606853, 1893139948,
      1583588576, 1219225407,
      2102034391, 1394921405,
      2005037790, 2031006861,
      1244218766, 923231061,
      49312790, 775496649,
      721012176, 321339902,
      1719909107, 1865748178,
      1156177430, 1257110891,
      307561322, 1918244397,
      906041433, 360476981,
      1591375755, 268492659,
      461522398, 227343256,
      2145930725, 2020665454,
      1938419274, 1331283701,
      174405412, 524140103,
      494343653, 18063908,
      1025534808, 181709577,
      2048959776, 1913665637,
      950636517, 794796256,
      1828843197, 1335757744,
      211109723, 983900607,
      825474095, 1046009991,
      374915657, 381856628,
      1241296328, 698149463,
      1260624655, 1024538273,
      900676210, 1628865823,
      697951025, 500570753,
      1007920268, 1708398558,
      264596520, 624727803,
      1977924811, 674673241,
      1440257718, 271184151,
      1928778847, 993535203,
      1307807366, 1801502463,
      1498732610, 300876954,
      1617712402, 1574250679,
      1261800762, 1556667280,
      949929273, 560721070,
      1766170474, 1953522912,
      1849939248, 19435166,
      887262858, 1219627824,
      483086133, 603728993,
      1330541052, 1582596025,
      1850591475, 723593133,
      1431775678, 1558439000,
      922493739, 1356554404,
      1058517206, 948567762,
      709067283, 1350890215,
      1044787723, 2144304941,
      999707003, 513837520,
      2140038663, 1850568788,
      1803100150, 127574047,
      867445693, 1149173981,
      408583729, 914837991,
      1166715497, 602315845,
      430738528, 1743308384,
      1388022681, 1760110496,
      1664028066, 654300326,
      1767741172, 1338181197,
      1625723550, 1742482745,
      464486085, 1507852127,
      754082421, 1187454014,
      1315342834, 425995190,
      960416608, 2004255418,
      1262630671, 671761697,
      59809238, 103525918,
      1205644919, 2107823293,
      1615183160, 1152411412,
      1024474681, 2118672937,
      1703877649, 1235091369,
      1821417852, 1098463802,
      1738806466, 1529062843,
      620780646, 1654833544,
      1070174101, 795158254,
      658537995, 1693620426,
      2055317555, 508053916,
      1647371686, 1282395762,
      29067379, 409683067,
      1763495989, 1917939635,
      1602690753, 810926582,
      885787576, 513818500,
      1853512561, 1195205756,
      1798585498, 1970460256,
      1819261032, 1306536501,
      1133245275, 37901,
      689459799, 1334389069,
      1730609912, 1854586207,
      1556832175, 1228729041,
      251375753, 683687209,
      2083946182, 1763106152,
      2142981854, 1365385561,
      763711891, 1735754548,
      1581256466, 173689858,
      2121337132, 1247108250,
      1004003636, 891894307,
      569816524, 358675254,
      626626425, 116062841,
      632086003, 861268491,
      1008211580, 779404957,
      1134217766, 1766838261,
      1423829292, 1706666192,
      942037869, 1549358884,
      1959429535, 480779114,
      778311037, 1940360875,
      1531372185, 2009078158,
      241935492, 1050047003,
      272453504, 1870883868,
      390441332, 1057903098,
      1230238834, 1548117688,
      1242956379, 1217296445,
      515648357, 1675011378,
      364477932, 355212934,
      2096008713, 1570161804,
      1409752526, 214033983,
      1288158292, 1760636178,
      407562666, 1265144848,
      1071056491, 1582316946,
      1014143949, 911406955,
      203080461, 809380052,
      125647866, 1705464126,
      2015685843, 599230667,
      1425476020, 668203729,
      1673735652, 567931803,
      1714199325, 181737617,
      1389137652, 678147926,
      288547803, 435433694,
      200159281, 654399753,
      1580828223, 1298308945,
      1832286107, 169991953,
      182557704, 1046541065,
      1688025575, 1248944426,
      1508287706, 1220577001,
      36721212, 1377275347,
      1968679856, 1675229747,
      279109231, 1835333261,
      1358617667, 1416978076,
      740626186, 2103913602,
      1882655908, 251341858,
      648016670, 1459615287,
      780255321, 154906988,
      857296483, 203375965,
      1631676846, 681204578,
      1906971307, 1623728832,
      1541899600, 1168449797,
      1267051693, 1020078717,
      1998673940, 1298394942,
      1914117058, 1381290704,
      426068513, 1381618498,
      139365577, 1598767734,
      2129910384, 952266588,
      661788054, 19661356,
      1104640222, 240506063,
      356133630, 1676634527,
      242242374, 1863206182,
      957935844, 1490681416};

  private RandomSeedTable() {
  }

  /**
   * Returns a deterministic seed from a (seemingly gigantic) matrix of predefined seeds.
   *
   * @param row    should (but need not) be in [0,Integer.MAX_VALUE].
   * @param column shoould (but need not) be in [0,SeedTable.COLUMNS-1].
   * @return the seed at the indicated matrix position.
   */
  public static int getSeedAtRowColumn(int row, int column) {
    // the table is limited; let's snap the unbounded input parameters to the table's actual size.
    int rows = rows();

    int theRow = Math.abs(row % rows);
    int theColumn = Math.abs(column % COLUMNS);

    int seed = seeds[theRow * COLUMNS + theColumn];

    // "randomize" the seed (in some ways comparable to hash functions)
    int cycle = Math.abs(row / rows);
    int mask = ((cycle & 0x000007ff) << 20); // cycle==0 --> mask = 0
    seed ^= mask;  // cycle==0 --> seed stays unaffected
    // now, each sequence has a period of 10**9 numbers.

    return seed;
  }

  /**
   * Not yet commented.
   *
   * @return int
   */
  private static int rows() {
    return seeds.length / COLUMNS;
  }
}
