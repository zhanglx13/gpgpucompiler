package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/**
 * Data-dependence Graph to store the result of dependence testing
 *
 */
public class DDGraph 
{
	public static final boolean summarize = true;
	public static final boolean not_summarize = false;
	
	private boolean summarized_status;
	
	private ArrayList<Arc> graphArcs;
	
	static public class Arc
	{
		/* source node */
		private DDArrayAccessInfo source;
		/* sink node */
		private DDArrayAccessInfo sink;
		/* byte depType
		 * 1 - Flow (True) Dependence
		 * 2 - Anti Dependence
		 * 3 - Output Dependence
		 * 4 - Input Dependence
		 */
		private byte depType;
		/* single direction vector from source to sink */
		DependenceVector directionVector;
		
		/**
		 * Creates a dependence arc from source expr1 to sink expr2 with the relevant direction vector
		 * @param expr1 - Contains all information related to source array access
		 * @param expr2 - Contains all information related to sink array access
		 * @param directionVector
		 */
		public Arc(DDArrayAccessInfo expr1, DDArrayAccessInfo expr2, DependenceVector directionVector)
		{
			
			if (directionVector.plausibleVector())
			{
				this.source = expr1;
				this.sink = expr2;
				this.directionVector = directionVector;
				setDependenceType(expr1.getAccessType(), expr2.getAccessType());
			}
			else
			{
				this.source = expr2;
				this.sink = expr1;
				this.directionVector = directionVector.reverseVector();
				setDependenceType(expr2.getAccessType(), expr1.getAccessType());
			}
		}
		
		public Arc(Arc a)
		{
			this.source = a.source;
			this.sink = a.sink;
			this.directionVector = new DependenceVector(a.directionVector);
			this.depType = a.depType;
		}
		
		public DDArrayAccessInfo getSource()
		{
			return this.source;
		}
		
		public DDArrayAccessInfo getSink()
		{
			return this.sink;
		}
		
		public Statement getSourceStatement()
		{
			return this.source.getParentStatement();
		}
		
		public Statement getSinkStatement()
		{
			return this.sink.getParentStatement();
		}
		
		public byte getDependenceType()
		{
			return this.depType;
		}
		
		public DependenceVector getDirectionVector()
		{
			return this.directionVector;
		}
		
		public void setDependenceType(int source_type, int sink_type)
		{
			if (source_type == DDArrayAccessInfo.write_type)
			{
				/* Output Dependence */
				if (sink_type == DDArrayAccessInfo.write_type)
					this.depType = 3;
				/* Flow Dependence */
				else if (sink_type == DDArrayAccessInfo.read_type)
					this.depType = 1;
			}
			else if(source_type == DDArrayAccessInfo.read_type)
			{
				/* Anti Dependence */
				if(sink_type == DDArrayAccessInfo.write_type)
					this.depType = 2;
				/* Don't test for Input Dependence */
			}
		}
		
		public String toString()
		{
			return ("ArcSource: " + this.source + " ArcSink: " + this.sink +
					" depType: " + Byte.valueOf(this.depType).toString() +
					" directionVector: " + this.directionVector.VectorToString()
					);
		}

	}
	
	public DDGraph ()
	{
		/* Initialize list of arcs in this graph */
		graphArcs = new ArrayList<Arc>();
		/* Arcs are not summarized, all dependences are explicit by default */
		summarized_status = false;
	}
	
	public void addArc(Arc arc)
	{
		graphArcs.add(arc);
	}

	public void deleteArc(Arc arc)
	{
		graphArcs.remove(arc);
	}
	
	/**
	 * Removes arcs with directions:
	 * (.) --> containing '.' = invalid merged direction
	 */
	private void filterUnwantedArcs()
	{
		Iterator<Arc> iter = graphArcs.iterator();

		while (iter.hasNext())
		{
			Arc arc = iter.next();
			DependenceVector dv = arc.getDirectionVector();

			if (!(dv.valid))
				//this.deleteArc(arc);
				// Cannot use this.deleteArc due to ArrayList
				// Iterator synchronization issues
				iter.remove();
			//else if ((dv.vector).containsValue(DependenceVector.any))
				//this.deleteArc(arc);
				// Cannot use this.deleteArc due to ArrayList
				// Iterator synchronization issues
				//iter.remove();
			else
			{
			}
		}
	}
	
	/**
	 * Filter out duplicate and unwanted arcs from the graph
	 */
	public void removeDuplicateArcs()
	{
		ListIterator<Arc> arc_list_1 = graphArcs.listIterator();
		while (arc_list_1.hasNext())
		{
			Arc arc1 = arc_list_1.next();
			int index = graphArcs.indexOf(arc1);

			if (arc1.getDirectionVector().valid)
			{
				ListIterator<Arc> arc_list_2 = graphArcs.listIterator(++index);
				while (arc_list_2.hasNext())
				{
					Arc arc2 = arc_list_2.next();
					if (arc2.getDirectionVector().valid)
					{
						if ((arc1.getSource() == arc2.getSource()) &&
								(arc1.getSink() == arc2.getSink()) &&
								(arc1.getDependenceType() == arc2.getDependenceType()) &&
								((arc1.getDirectionVector().vector).equals(arc2.getDirectionVector().vector)))
							arc2.getDirectionVector().valid = false;
					}
				}
			}
		}
		
		this.filterUnwantedArcs();
	}

	/**
	 * Summarize the direction vectors between nodes of this graph
	 */
	public void summarizeGraph()
	{
		this.summarized_status = true;
		/* Create a new set of arcs that must be added to the graph */
		ArrayList<Arc> newArcsForGraph = new ArrayList<Arc>();
		
		Iterator<Arc> arc_list = graphArcs.iterator();
		
		while (arc_list.hasNext())
		{
			Arc arc = arc_list.next();
			
			if (arc.getDirectionVector().valid)
			{
				Set<Loop> loopsInVector = ((arc.getDirectionVector())).getLoops();
				ArrayList<Arc> arcsToBeSummarized = new ArrayList<Arc>();
				arcsToBeSummarized = getDependenceArcsFromTo(arc.getSource().getArrayAccess(),
						arc.getSink().getArrayAccess());

				ArrayList<DependenceVector> dvsToBeSummarized = new ArrayList<DependenceVector>();

				for (Arc arc_iter : arcsToBeSummarized)
				{
					dvsToBeSummarized.add(arc_iter.getDirectionVector());
				}

				for (Loop l : loopsInVector)
				{
					int equal_dir_cnt = 0;
					int less_dir_cnt = 0;
					int great_dir_cnt = 0;
					for (DependenceVector v : dvsToBeSummarized)
					{
						switch (v.getDirection(l))
						{
						case DependenceVector.equal:
							equal_dir_cnt++;
							break;
						case DependenceVector.less:
							less_dir_cnt++;
							break;
						case DependenceVector.greater:
							great_dir_cnt++;
							break;
						default:
							break;
						}
					}
					/* Check if all directions are present for that loop, then we can summarize */
					if ((equal_dir_cnt > 0) &&
							(less_dir_cnt > 0) &&
							(great_dir_cnt > 0))
					{
						for (Arc a :  arcsToBeSummarized)
						{
							Arc summarized_arc = new Arc(a);
							(summarized_arc.getDirectionVector()).setDirection(l, DependenceVector.any);
							newArcsForGraph.add(summarized_arc);
							/* Invalidate the old arc, it will be deleted by the remove duplicates routine */
							a.getDirectionVector().valid = false;
						}
					}
				}
			}
		}
		
		/* Add all new arcs to the graph */
		for (Arc arc : newArcsForGraph)
		{
			graphArcs.add(arc);
		}
		
		/* Remove duplicate arcs that might have been added as part of the summarization process */
		this.removeDuplicateArcs();
	}
	
	/**
	 * Return true if there are any loop carried dependences within arrays of this graph
	 * @return
	 */
	public boolean checkLoopCarriedDependenceForGraph()
	{
		Iterator<Arc> iter = graphArcs.iterator();

		while (iter.hasNext())
		{
			Arc arc = iter.next();
			DependenceVector dv = arc.getDirectionVector();

			if ((dv.vector).containsValue(DependenceVector.less) ||
					(dv.vector).containsValue(DependenceVector.greater) ||
					(dv.vector).containsValue(DependenceVector.any))
				return true;
		}
		return false;
	}

	/**
	 * Check for equal dependences at the level given by parameter 'loop'
	 * @param loop
	 * @return
	 */
	public boolean checkEqualDependences(Loop loop)
	{
		Iterator<Arc> iter = graphArcs.iterator();
		
		while (iter.hasNext())
		{
			Arc arc = iter.next();
			DependenceVector dv = arc.getDirectionVector();
			
			if (((dv.vector).get(loop)) != DependenceVector.equal)
				return false;
		}
		return true;
	}
	
	/**
	 * Check whether a loop is parallel or not using the dependence graph
	 * @param loop
	 * @return
	 */
	public boolean checkParallel(Loop loop)
	{
		return (this.checkEqualDependences(loop));
	}
	
	/**
	 * Obtain all possible dependence information from expr1 to expr2 in a given loop
	 * @param expr1 - ArrayAccess
	 * @param expr2 - ArrayAccess
	 * @return arcSet - List of all existing dependence arcs from expr1 to expr2
	 */
	public ArrayList<Arc> getDependenceArcsFromTo(ArrayAccess expr1, ArrayAccess expr2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();
		
		for(Arc arc: graphArcs)
		{
			if (arc.getSource().getArrayAccess() == expr1)
			{
				if (arc.getSink().getArrayAccess() == expr2)
					arcSet.add(arc);
			}
		}
		return (arcSet);
	}
	
	/**
	 * Obtain all possible dependence information between a pair of array accesses in a given loop
	 * @param expr1 - ArrayAccess
	 * @param expr2 - ArrayAccess
	 * @return arcSet - List of all existing dependence arcs between the two accesses
	 */
	public ArrayList<Arc> getDependenceArcsBetween(ArrayAccess expr1, ArrayAccess expr2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();

		arcSet.addAll(getDependenceArcsFromTo(expr1, expr2));

		arcSet.addAll(getDependenceArcsFromTo(expr2, expr1));

		return (arcSet);
	}
	
	/**
	 * Obtain all possible dependence information between a pair of statements in a given loop
	 * @param stmt1 - Statement
	 * @param stmt2 - Statement
	 * @return arcSet - List of all existing dependence arcs between the two statements
	 */
	public ArrayList<Arc> getDependenceArcsBetween(Statement stmt1, Statement stmt2)
	{
		ArrayList<Arc> arcSet = new ArrayList<Arc>();
		
		for(Arc arc: graphArcs)
		{
			if (arc.getSourceStatement() == stmt1)
			{
				if (arc.getSinkStatement() == stmt2)
					arcSet.add(arc);
			}
			else if (arc.getSourceStatement() == stmt2)
			{
				if (arc.getSinkStatement() == stmt1)
					arcSet.add(arc);
			}
		}
		return (arcSet);
	}
	
	public String toString()
	{
		String temp = "Arc Info for this DD graph\n";
		
		for (Arc arc: graphArcs)
		{
			temp += (arc.toString() + "\n");
		}
		return temp;
	}
	
}
