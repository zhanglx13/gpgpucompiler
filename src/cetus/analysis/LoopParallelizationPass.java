package cetus.analysis;

import java.util.*;

import cetus.hir.*;

/**
 * Whole program analysis that uses data-dependence information to 
 * internally annotate loops that are parallel
 */
public class LoopParallelizationPass extends AnalysisPass
{
	public LoopParallelizationPass(Program program)
	{
		super(program);
	}

	/**
	 * Get Pass name
	 */
	public String getPassName()
	{
		return new String("[LOOP-PARALLELIZATION]");
	}
	
	/**
	 * Start whole program loop parallelization analysis
	 */
	public void start()
	{
		DepthFirstIterator dfs_iter = new DepthFirstIterator(program);
		Loop current_outermost_loop = null;
		DDGraph loop_graph = null;
		
		for (;;)
		{
			Loop loop = null;
			try {
				loop = (Loop)dfs_iter.next(Loop.class);
			}
			catch (NoSuchElementException e) {
				break;
			}
			
			/* Run dependence analysis on the nest if this is the outermost loop and obtain the
			 * dependence graph
			 */
			if(LoopTools.isOutermostLoop(loop))
			{
				current_outermost_loop = loop;
				//loop_graph = (new DDTDriver(program)).getDDGraph(DDGraph.not_summarize, current_outermost_loop);
				loop_graph = (new DDTDriver(program)).getDDGraph(DDGraph.summarize, current_outermost_loop);
			}
			
			/* Graph could be null if loop nest was not eligible for dependence analysis */
			if (loop_graph != null)
			{
				/* Check loop_graph for direction vectors pertaining to each loop and decide
				 * which loops in this nest are parallel */
				boolean no_dependence = loop_graph.checkParallel(loop);
				addAnnotation(loop, no_dependence);
			}
		}
	}
	
	private void addAnnotation(Loop loop, boolean parallel)
	{
		if (parallel)
		{
			Statement stmt = (Statement)loop;
			Annotation note = new Annotation("cetus");
			note.setPrintMethod(Annotation.print_as_pragma_method);
			note.add("parallel", "true");
			AnnotationStatement astmt = new AnnotationStatement(note);
			astmt.attachStatement(stmt);
			CompoundStatement parent = (CompoundStatement)stmt.getParent();
			parent.addStatementBefore(stmt, astmt);
		}
		//else
		//note.add("parallel", "false");
	}
	
}
