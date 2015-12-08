package cetus.analysis;
import java.util.*;
import cetus.hir.*;

/**
 * Creates a pair of affine subscripts where subscript is a single dimension of an array reference
 */
public class SubscriptPair {
	
	/* Store normalized expression if affine */
	private NormalExpression normalSubscript1, normalSubscript2;
	/* Loops from which indices are present in the subscript pair */
	private LinkedList<Loop> present_loops;
	/* Loop indices present in the affine expressions */
	private List<Identifier> ids_in_expressions;
	/* All loops from the enclosing loop nest */
	private LinkedList<Loop> enclosing_loops;
	/* Loop information for the enclosing loop nest */
	private HashMap<Loop, LoopInfo> enclosing_loops_info;
	/* Affine or not */
	boolean is_affine;
	
	public SubscriptPair (Expression subscript1, Expression subscript2, LinkedList<Loop> nest, HashMap <Loop,LoopInfo> loopinfo)
	{
		/* Obtain the normalized subscript expressions and store if they're affine */
		normalSubscript1 = NormalExpression.normalSimplify(subscript1);
		normalSubscript2 = NormalExpression.normalSimplify(subscript2);
		
		/* Check for affine subscripts using the NormalExpression affine check */
		LinkedList<Identifier> id_list = new LinkedList<Identifier>();
		for (Loop l: nest)
		{
			id_list.add((Identifier)LoopTools.getIndexVariable(l));
		}
		
		if ((normalSubscript1.isAffine(id_list)) &&
				(normalSubscript2.isAffine(id_list)))
			this.is_affine = true;
		else
			this.is_affine = false;
		
		if (is_affine)
		{
			ids_in_expressions = normalSubscript1.getVariableList();
			ids_in_expressions.addAll(normalSubscript2.getVariableList());

			this.enclosing_loops = nest;
			this.enclosing_loops_info = loopinfo;
			present_loops = new LinkedList<Loop>();		
			for (Loop loop: nest)
			{
				LoopInfo info = loopinfo.get(loop);
				Expression index = info.indexVar;
				if (ids_in_expressions.contains(index))
				{
					present_loops.addLast(loop);
				}
			}
		}
	}
	
	HashMap<Loop,LoopInfo> getEnclosingLoopsInfo()
	{
		return enclosing_loops_info;
	}
	
	LinkedList<Loop> getEnclosingLoopsList()
	{
		return enclosing_loops;
	}
	
	LinkedList<Loop> getPresentLoops()
	{
		return present_loops;
	}
	
	NormalExpression getSubscript1()
	{
		return normalSubscript1;
	}
	NormalExpression getSubscript2()
	{
		return normalSubscript2;
	}
	int getComplexity()
	{
		return present_loops.size();
	}
}
