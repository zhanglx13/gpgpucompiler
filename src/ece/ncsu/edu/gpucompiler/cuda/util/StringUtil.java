package ece.ncsu.edu.gpucompiler.cuda.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class StringUtil {
//	public static void output(Traversable ts, int i) {
//		System.out.print(i+":");
//		System.out.println(ts.getClass());
//		System.out.println(ts.toString());
//		List<Traversable> list = (List<Traversable>)ts.getChildren();
//		if (list!=null) for (Traversable s: list) output(s, i+1);
//	}
//	
//
//	
	public static String toString(Stack<String> stack) {
		StringBuffer sb = new StringBuffer();
		for (int i=0; i<stack.size(); i++) {
			sb.append((i==0?"":".")+stack.get(i));
		}
		return sb.toString();
	}

	public static void main(String[] args) {
		//int nbid = (bid/2/stride)*stride*2+(bid%stride);
		//nbid = nbid + (cbid*stride);
		int stride = 64;
		for (int i=0; i<128; i++) {
			int bid = i;
			int cbid = bid&1;
			int nbid = (bid/2/stride)*stride*2+((bid/2)%stride);
			nbid = nbid + (cbid*stride);
			System.out.print(nbid+",");
		}
	}
	
	
	public static List<String> subtract(List<String> a, List<String> b) {
		List<String> result = new ArrayList();
		for (int i=0; i<a.size(); i++) {
			String s = a.get(i);
			if (!b.contains(s)) {
				result.add(s);
			}
		}
		return result;
	}
}
