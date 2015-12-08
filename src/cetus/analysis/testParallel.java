package cetus.analysis;

import cetus.hir.*;
import cetus.exec.*;

/**
	* This pass analyzes openmp pragmas and converts them into Cetus Annotations, 
	* the same form as what parallelization passes generate. The original OpenMP 
	* pragmas are removed after the analysis.
	*/
public class testParallel extends AnalysisPass
{
	private int debug_level;

	public testParallel(Program program)
	{
		super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
	}

	public String getPassName()
	{
		return new String("[testParallel]");
	}

	public void start()
	{

		/* Analyze OpenMP input program to generate omp annotations */
		OmpAnalysis omp = new OmpAnalysis(program);
		AnalysisPass.run(omp);

		/* Analyze OpenMP input program to generate omp annotations */
		LoopParallelizationPass loopParPass = new LoopParallelizationPass(program);
		AnalysisPass.run(loopParPass);

		/* iterate over everything, with particular attention to annotations */
		DepthFirstIterator iter = new DepthFirstIterator(program);

		while(iter.hasNext())
		{
		}
	}
}

	
