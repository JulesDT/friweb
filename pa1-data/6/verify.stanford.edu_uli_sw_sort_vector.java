generic bubble sort in java vector version written jan 00 by ulrich stern import java util for vector class interface comparator boolean compare object o1 object o2 class lib static void sort vector a comparator cmp for int i 0 i a size 1 i++ for int j a size 1 j i j if cmp compare a elementat j 1 a elementat j object tmp a elementat j 1 a setelementat a elementat j j 1 a setelementat tmp j public class sort_vector static class range int lb int ub range int l int u lb l ub u public string tostring return lb ub compare lower bound static class lbcomparator implements comparator public boolean compare object o1 object o2 return range o1 lb range o2 lb compare upper bound static class ubcomparator implements comparator public boolean compare object o1 object o2 return range o1 ub range o2 ub public static void main string args final int n 10000 vector a new vector n a setsize n for int i 0 i a size i++ a setelementat new range i 7 1000 i 13 1000 i system out println sorting lib sort a new lbcomparator lib sort a new ubcomparator system out println done