package ece.ncsu.edu.gpucompiler.cuda.pass.simple;

import java.util.List;

import cetus.hir.AssignmentExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Procedure;
import cetus.hir.Traversable;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.pass.Pass;

public class MemoryExpressionPass extends Pass {

	
	
	public MemoryExpressionPass() {
		
	}
	
	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	@Override
	public void dopass(GProcedure proc) {
		int id = 0;
		Procedure procedure = proc.getProcedure();
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		List<AssignmentExpression> assignmentExpressions = (List<AssignmentExpression>)dfi.getList(AssignmentExpression.class);
		for (AssignmentExpression assignmentExpression: assignmentExpressions) {
			if (!MemoryExpression.isMemoryExpression(assignmentExpression, proc)) continue;
			try {
				Traversable tr = assignmentExpression.getParent();
				while (!(tr instanceof GLoop)&&tr!=null) {
					tr = tr.getParent();
				}
				if (tr!=null) {
					GLoop loop = (GLoop)tr;
					MemoryExpression me;
					me = new MemoryExpression(assignmentExpression, loop);
					me.setId(id);
					id++;
					assignmentExpression.swapWith(me);
				}
			} catch (UnsupportedCodeException e) {
				e.printStackTrace();
			}
		}
		
	}
}
