package ece.ncsu.edu.gpucompiler.cuda.merger;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedureUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;

public class ThreadMerger extends Merger {


	
	public void merge(GProcedure proc) {

		if (getMergeAction().getMergeDirection()==Merger.MERGE_DIRECTION_Y) {
			
			// coalesced_idy -> coalesced_idy/ (merge number) + it*(merge number)+tidy
			/*
			 * 
			 * coalesced_idy = bidy*16 
			 * after thread merge 8 into together
			 * a[coalesced_idy]
			 * ->
			 * for (int i=0; i<8; i++) a[8*bidy + it*2 + tidy]
			 * 
			 */
			
			int maxNumber = getMergeAction().getMergeNumber();

			List<MemoryExpression> memoryExpressions = new ArrayList();
			for (BlockDiff diff: getMergeAction().getBlockDiffs()) {
				memoryExpressions.add(diff.getMemoryExpression());
				if (diff.getMemoryExpressionR()!=null) memoryExpressions.add(diff.getMemoryExpressionR());
				
			}			

			
			List<VariableDeclaration> decsTodo = new ArrayList();
			List<ExpressionStatement> essTodo = new ArrayList();
			
			GProcedureUtil.loadTodos(proc, memoryExpressions, decsTodo, essTodo);

			
			
			BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.BIDY), BinaryOperator.MULTIPLY, new IntegerLiteral(maxNumber*proc.getBlockDimY()));
			be =  new BinaryExpression(be, BinaryOperator.ADD, new Identifier(ThreadIndex.TIDY));

			GProcedureUtil.duplicateDeclaration(proc, decsTodo, maxNumber);
			GProcedureUtil.duplicateStatementWithShare(proc, decsTodo, essTodo, maxNumber, false, proc.getBlockDimY());
			
			CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDY), be);
		}
		else {
			// not tested yet
			/*
			int maxNumber = getMergeAction().getMergeNumber();

			List<MemoryExpression> memoryExpressions = new ArrayList();
			for (BlockDiff diff: getMergeAction().getBlockDiffs()) {
				memoryExpressions.add(diff.getMemoryExpression());
				if (diff.getMemoryExpressionR()!=null) memoryExpressions.add(diff.getMemoryExpressionR());
				
			}			

			
			List<VariableDeclaration> decsTodo = new ArrayList();
			List<ExpressionStatement> essTodo = new ArrayList();
			
			GProcedureUtil.loadTodos(proc, memoryExpressions, decsTodo, essTodo);

			
			
			BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.MULTIPLY, new IntegerLiteral(maxNumber*proc.getBlockDimX()));
			be =  new BinaryExpression(be, BinaryOperator.ADD, new Identifier(ThreadIndex.TIDX));

			GProcedureUtil.duplicateDeclaration(proc, decsTodo, maxNumber);
			GProcedureUtil.duplicateStatementWithShare(proc, decsTodo, essTodo, maxNumber, true, proc.getBlockDimY());
			
			CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDX), be);		
			*/	
		}
	}
	

}
