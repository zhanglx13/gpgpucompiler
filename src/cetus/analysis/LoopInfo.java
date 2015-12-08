package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/**
 * Represents loop-related information
 */
public class LoopInfo 
{
	//private Expression upperBound; /*IntLiteral or Variable*/
	public long upperBound;
	//private Expression lowerBound; /*IntLiteral or Variable*/
	public long lowerBound;
	//private Expression increment; /*IntLiteral or Variable*/
	public long increment;
	
	public Expression indexVar;
	
	LinkedList<Loop> loopNest; /*set of all enclosing outermost loops and the loop itself*/
	
	public LoopInfo ()
	{
		this.upperBound = 0;
		this.lowerBound = 0;
		this.increment = 0;
		this.indexVar = null;
		this.loopNest = null;
	}
	
	/**
	 * Creates a data structure containing loop-related information (use only if canonical loop)
	 * @param loop
	 */
	public LoopInfo (Loop loop)
	{
		this.upperBound = LoopTools.getUpperBound(loop);
		this.lowerBound = LoopTools.getLowerBound(loop);
		this.increment = LoopTools.getIncrement(loop);
		this.indexVar = LoopTools.getIndexVariable(loop);
		this.loopNest = LoopTools.calculateLoopNest(loop);
	}
	
	public String toString()
	{
		return new String(indexVar.toString() +
				" from " + Long.toString(lowerBound) +
				" to "   + Long.toString(upperBound) +
				" step " + Long.toString(increment));
	}
	
}
