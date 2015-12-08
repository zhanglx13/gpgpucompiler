package cetus.analysis;
import java.util.*;

import cetus.hir.*;

/**
 * Stores and manipulates direction vectors for loop-based dependences
 *
 */
public class DependenceVector {
	String[] depstr = {"*","<","=",">"};
	
	static final int nil = -1;
	static final int any = 0;
	static final int less = 1;
	static final int equal = 2;
	static final int greater = 3;
	int cartesian_prod[][]={
			{DependenceVector.any, DependenceVector.less, DependenceVector.equal, DependenceVector.greater},
			{DependenceVector.less, DependenceVector.less, DependenceVector.nil, DependenceVector.nil},
			{DependenceVector.equal, DependenceVector.nil, DependenceVector.equal,DependenceVector.nil},
			{DependenceVector.greater, DependenceVector.nil, DependenceVector.nil, DependenceVector.greater}
			};

	/* LinkedHashMap maintains ordering of loops within the vector map */
	LinkedHashMap<Loop,Integer> vector;
	boolean valid = true;
	
	void mergeWith (DependenceVector other_vector)
	{
		int new_dir;
		for (Loop l : other_vector.getLoops())
		{
			if (vector.containsKey(l))
			{
				int this_dir = this.getDirection(l);
				int that_dir = other_vector.getDirection(l);
				if (this_dir != DependenceVector.nil)
				{
					new_dir = cartesian_prod[this_dir][that_dir];
				}
				else
				{
					new_dir = DependenceVector.nil;
				}
				if (new_dir == DependenceVector.nil) valid = false;
				this.vector.put(l, new_dir);
			}
			else
			{
				this.vector.put(l, other_vector.getDirection(l));
			}
		}
	}
	
	public DependenceVector(LinkedList <Loop> nest)
	{
		vector = new LinkedHashMap<Loop,Integer>();
		for (Loop loop: nest)
		{
			vector.put(loop, 0); //any value
		}
	}
	
	public DependenceVector(DependenceVector dv)
	{
		this.copyVector (dv);
	}
	
	int getDirection (Loop loop)
	{
		return vector.get(loop);
	}
	
	Set<Loop> getLoops()
	{
		return vector.keySet();
	}
	
	void setDirection (Loop loop, int direction)
	{
		vector.put(loop,direction);
	}
	
	void copyVector (DependenceVector dv)
	{
		this.valid = dv.valid;
		
		vector = new LinkedHashMap<Loop,Integer>();
		for (Loop loop: dv.getLoops())
		{
			vector.put(loop, dv.getDirection(loop));
		}
	}
	
	public boolean plausibleVector()
	{
		boolean vectorValid = true;
		Set <Loop> loopNest = this.vector.keySet();
		
		for (Loop loop : loopNest)
		{
			/*
			 * Following invalid possibilities:
			 * (>,...) , (=,=,>,...) , (*,>,...)
			 */
			if (this.vector.get(loop) == DependenceVector.greater)
			{
				vectorValid = false;
				break;
			}
			/*
			 * Else if following valid possibilities:
			 * (<,...) , (=,<,...) , (*,<,...)
			 */
			else if (this.vector.get(loop) == DependenceVector.less)
			{
				vectorValid = true;
				break;
			}
			/*
			 * Else we need to further traverse the vector
			 * (=,...) , (*,...)
			 */
			else
			{
			}
		}
		
		return vectorValid;
	}
	
	public DependenceVector reverseVector()
	{
		DependenceVector newDV = new DependenceVector(this);
		
		Set<Loop> loopKey = (this.vector).keySet();
		
		for(Loop loop : loopKey)
		{
			switch(this.vector.get(loop))
			{
			case DependenceVector.any:
			case DependenceVector.equal:
			case DependenceVector.nil:
				break;
				
			case DependenceVector.less:
				newDV.setDirection(loop, DependenceVector.greater);
				break;
				
			case DependenceVector.greater:
				newDV.setDirection(loop, DependenceVector.less);
				break;
			}
		}
		return newDV;
	}
	
	String VectorToString ()
	{
		if (this.valid)
		{
			String dirvecstr = new String();
			Set<Loop> nest = this.vector.keySet();
			for (Loop loop: nest)
			{
				dirvecstr += this.depstr[vector.get(loop)];
			}
			return dirvecstr;
		}
		else
		{
			return ".";
		}
	}
}
