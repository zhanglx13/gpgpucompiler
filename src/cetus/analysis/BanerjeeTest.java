package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/**
 * Banerjee Test implements data-dependence testing for a pair of
 * affine subscripts using Banerjee inequalities
 */
public class BanerjeeTest {
	long getPositivePart (long a)
	{
		if (a >= 0)
			return a;
		else
			return 0;
	}
	long getNegativePart (long a)
	{
		if (a <= 0)
			return a;
		else
			return 0;
	}
	
	static String[] depstr = {"*","<","=",">"};
	static final int LB = 0;
	static final int UB = 4;
	
	static final int nil = -1;
	static final int any = 0;
	static final int less = 1;
	static final int equal = 2;
	static final int greater = 3;
	
	static final int LB_any = 0;
	static final int LB_less = 1;
	static final int LB_equal = 2;
	static final int LB_greater = 3;
	
	static final int UB_any = 4;
	static final int UB_less = 5;
	static final int UB_equal = 6;
	static final int UB_greater = 7;
	
	/* Stores bounds calculated as per banerjee's inequalities for every loop
	 * in the nest
	 */
	HashMap<Loop, Vector<Long>> banerjee_bounds;
	
	/* Constant coefficients in the affine subscript expressions as identified
	 * by normalising the expression
	 */
	long const1, const2;
	
	/**
	 * Constructor
	 * @param normalSubscript1
	 * @param normalSubscript2
	 * @param loopnest - Store loop nest ArrayList<Loop>
	 * @param loopInfo - Provide loop information for all loops in the nest
	 */
	public BanerjeeTest (NormalExpression normalSubscript1, 
				  NormalExpression normalSubscript2, 
				  LinkedList<Loop> loopnest,
				  HashMap<Loop, LoopInfo>loopInfo
				  )
	{
		banerjee_bounds = new HashMap<Loop,Vector<Long>>();
		
		List<Identifier> idlist = new ArrayList<Identifier>();
		
		for (Loop loop_id : loopnest)
		{
			Identifier index = (Identifier)(loopInfo.get(loop_id)).indexVar;
			idlist.add(index);
		}

		this.const1 = normalSubscript1.getConstantCoefficient();
		this.const2 = normalSubscript2.getConstantCoefficient();
		
		
		//For each LoopInfo object, compute the Banerjee bounds 
		//and add it to the map
		for (Loop loop: loopnest)
		{
				
			Identifier id = (Identifier)(loopInfo.get(loop)).indexVar;
			Tools.println("indexVariable " + id.getName(), 2);
			Vector<Long> bounds = new Vector<Long> (8); //Banerjee bounds
			
			
			long A = normalSubscript1.getCoefficient(id);
			Tools.println("normalSubscript1.getCoefficient(id) " + A, 2);
			long B = normalSubscript2.getCoefficient(id);
			Tools.println("normalSubscript2.getCoefficient(id) " + B, 2);

			long U = (loopInfo.get(loop)).upperBound;
			Tools.println("upperBound " + U, 2);
			long L = (loopInfo.get(loop)).lowerBound;
			Tools.println("lowerBound " + L, 2);
			long N = (loopInfo.get(loop)).increment;
			Tools.println("increment " + N, 2);
				
			bounds.add(BanerjeeTest.LB_any, new Long((getNegativePart(A)-getPositivePart(B))*(U-L)+(A-B)*L));
			bounds.add(BanerjeeTest.LB_less, new Long(getNegativePart(getNegativePart(A)-B)*(U-L-N)+(A-B)*L - B*N));
			bounds.add(BanerjeeTest.LB_equal, new Long(getNegativePart(A-B)*(U-L)+(A-B)*L));
			bounds.add(BanerjeeTest.LB_greater, new Long(getNegativePart (A-getPositivePart(B))*(U-L-N)+(A-B)*L+A*N));
			
			bounds.add(BanerjeeTest.UB_any, new Long((getPositivePart(A)-getNegativePart(B))*(U-L)+(A-B)*L));
			bounds.add(BanerjeeTest.UB_less, new Long(getPositivePart(getPositivePart(A)-B)*(U-L-N)+(A-B)*L - B*N));
			bounds.add(BanerjeeTest.UB_equal, new Long(getPositivePart(A-B)*(U-L)+(A-B)*L));
			bounds.add(BanerjeeTest.UB_greater, new Long(getPositivePart (A-getNegativePart(B))*(U-L-N)+(A-B)*L+A*N));
				
			banerjee_bounds.put(loop, bounds);
		}		
	}
	
	
	public boolean testDependenceVector (LinkedList<Loop> nest, DependenceVector dependence_vector)
	{
		long banerjeeLB=0;
		long banerjeeUB=0;
		
		long diff = (this.const2 - this.const1);
	
		for (Loop loop : nest)
		{
			int loop_dependence_direction = dependence_vector.getDirection(loop);
			banerjeeLB += (Long)banerjee_bounds.get(loop).get(loop_dependence_direction+BanerjeeTest.LB);
			banerjeeUB += (Long)banerjee_bounds.get(loop).get(loop_dependence_direction+BanerjeeTest.UB);
		}
		
		if (diff < banerjeeLB || diff > banerjeeUB)
		{
			Tools.println("Dependence does not exist", 2);
			printDirectionVector((dependence_vector.vector), nest);
			return false;
		}
		else
		{
			Tools.println("Dependence exists", 2);
			printDirectionVector((dependence_vector.vector), nest);
			return true;
		}
	}
	
	public void printDirectionVector(HashMap<Loop,Integer> DependenceVector, LinkedList<Loop> nest)
	{
		Tools.print("(", 2);
		for (int i=0; i< nest.size(); i++)
		{
			Loop loop = nest.get(i);
			Tools.print(BanerjeeTest.depstr[DependenceVector.get(loop)], 2);
		}
		Tools.println(")", 2);
	}
	
	void testTree(DependenceVector dv, int pos, LinkedList<Loop> nest, ArrayList<DependenceVector> dv_list)
	{
		
		//test for all the others
		for (int dir=BanerjeeTest.less; dir <= BanerjeeTest.greater; dir++)
		{
			Loop loop = nest.get(pos);
			dv.setDirection(loop, dir);
			if (testDependenceVector(nest, dv))
			{
				DependenceVector dv_clone = new DependenceVector(dv);
				/* Add to dependence vector list only if it does not contain
					the 'any' (*) direction for all given loops */
				if (!((dv_clone.vector).containsValue(DependenceVector.any)))
					dv_list.add(dv_clone);
				if ((pos+1) < nest.size())
					testTree(dv, pos+1, nest, dv_list);
			}
//			if ((pos+1) < nest.size())
//				testTree(dv, pos + 1, nest, dv_list);
			dv.setDirection(loop, BanerjeeTest.any);
		}
		return;
	}
	
	public ArrayList<DependenceVector> testAllDependenceVectors(LinkedList<Loop> nest)
	{
		ArrayList <DependenceVector> dv_list = new ArrayList<DependenceVector>();
		DependenceVector dv = new DependenceVector (nest); //create vector dv=(*,...,*);
			  
		/* add dependence vector to list if there is a (*,...,*) dependence*/
		if (testDependenceVector (nest, dv))
		{
			DependenceVector dv_clone = new DependenceVector (dv);
			/* Add to dependence vector list only if it does not contain
					the 'any' (*) direction for all given loops */
			if (!((dv_clone.vector).containsValue(DependenceVector.any)))
				dv_list.add(dv_clone);
			testTree(dv, 0, nest, dv_list);
		}
		//testTree(dv, 0, nest, dv_list);
		return dv_list;
	}
}

