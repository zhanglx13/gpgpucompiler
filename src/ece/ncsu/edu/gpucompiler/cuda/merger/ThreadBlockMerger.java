package ece.ncsu.edu.gpucompiler.cuda.merger;

import java.util.ArrayList;
import java.util.List;

import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Procedure;
import cetus.hir.Statement;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.DeclarationUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;


public class ThreadBlockMerger extends Merger {

	

	
	public void merge(GProcedure proc) {
		int coalescednumber = CudaConfig.getCoalescedThread();
		if (getMergeAction().getMergeDirection()==Merger.MERGE_DIRECTION_X) {
			// only the threads in first block only load it
			// tidx->(tidx%coalesceNumber)
			Procedure procedure = proc.getProcedure();
			DepthFirstIterator dfi = new DepthFirstIterator(procedure);
			List<MemoryExpression> memoryExpressions = dfi.getList(MemoryExpression.class);
			List<String> shared = new ArrayList();
			for (int i=0; i<memoryExpressions.size(); i++) {
				MemoryExpression me = memoryExpressions.get(i);
				MemoryArrayAccess maa = me.getGlobalMemoryArrayAccess();
				boolean needToHandle = false;
				if (maa!=null) {
					if (maa.getX().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) needToHandle = true;
					if (maa.getY()!=null&&maa.getX().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) needToHandle = true;
				}
				maa = me.getSharedMemoryArrayAccess();
				if (maa!=null) {
					if (maa.getX().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) needToHandle = true;
					if (maa.getY()!=null&&maa.getY().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) needToHandle = true;
				}
				if (!needToHandle) {
					memoryExpressions.remove(i);
					i--;						
				}
			}
			
			for (BlockDiff blockDiff: getMergeAction().getBlockDiffs()) {
				MemoryExpression memoryExpression = blockDiff.getMemoryExpression();
				MemoryExpression memoryExpressionR = blockDiff.getMemoryExpressionR();
				if (memoryExpression.getSharedMemoryArrayAccess()!=null) {
					shared.add(memoryExpression.getSharedMemoryArrayAccess().getArrayName().toString());
				}
				if (memoryExpressionR!=null&&memoryExpressionR.getSharedMemoryArrayAccess()!=null) {
					shared.add(memoryExpressionR.getSharedMemoryArrayAccess().getArrayName().toString());
				}
				for (int i=0; i<memoryExpressions.size(); i++) {
					MemoryExpression me = memoryExpressions.get(i);
					if (me.getId()==memoryExpression.getId()||
							(memoryExpressionR!=null&&me.getId()==memoryExpressionR.getId())) {
						memoryExpressions.remove(i);
						i--;
					}
				}
				if (memoryExpressionR==null) {
					// 
					BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), 
							BinaryOperator.COMPARE_LT, new IntegerLiteral(proc.getBlockDimX()));
					
					IfStatement ifs = new IfStatement(be, (Statement)memoryExpression.getStatement().clone());
					ifs.swapWith(memoryExpression.getStatement());
				}
				else {
					// 1. expand the size of shared memory
					// 2. first statement (if tidx<coalesceNumber), second statement no control flow
					BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), 
							BinaryOperator.COMPARE_LT, new IntegerLiteral(proc.getBlockDimX()));
					
					IfStatement ifs = new IfStatement(be, (Statement)memoryExpression.getStatement().clone());
					ifs.swapWith(memoryExpression.getStatement());
					
					DepthFirstIterator dfi_1 = new DepthFirstIterator(proc.getProcedure());
					List<VariableDeclaration> decs = dfi_1.getList(VariableDeclaration.class);
					for (VariableDeclaration ds: decs) {
						MemoryArrayAccess maa = memoryExpression.getSharedMemoryArrayAccess();
						if (DeclarationUtil.getVariableName(ds).equals(maa.getArrayName().toString())) {
							ArraySpecifier as = (ArraySpecifier)ds.getDeclarator(0).getArraySpecifiers().get(0);
							IntegerLiteral il = (IntegerLiteral)as.getDimension(0);
							as.setDimension(0, new IntegerLiteral(coalescednumber*getMergeAction().getMergeNumber()-coalescednumber+il.getValue()));
						}
					}
					
				}
			}
			
			for (int i=0; i<memoryExpressions.size(); i++) {
				MemoryExpression me = memoryExpressions.get(i);
				MemoryArrayAccess maa = me.getSharedMemoryArrayAccess();
				if (maa!=null) {
					if (shared.contains(maa.getArrayName().toString())) {
 						memoryExpressions.remove(i);
						i--;						
					}
				}
			}
			
			List<String> sharedTodo = new ArrayList();
			for (int i=0; i<memoryExpressions.size(); i++) {
				MemoryExpression me = memoryExpressions.get(i);
				CetusUtil.replaceChild(me, new Identifier(ThreadIndex.TIDX), new Identifier("ntidx"));
				if (me.getSharedMemoryArrayAccess()!=null) {
					MemoryArrayAccess maa = me.getSharedMemoryArrayAccess();
					if (shared.contains(maa.getArrayName().toString())) continue;
					maa.addIndex(new Identifier("null"));
					for (int k=maa.getNumIndices()-1; k>0; k--) {
						maa.setIndex(k, (Expression)maa.getIndex(k-1).clone());
					}
					maa.setIndex(0, new Identifier("ibidx"));
					if (!sharedTodo.contains(maa.getArrayName().toString()))
						sharedTodo.add(maa.getArrayName().toString());
				}
			}
			
			for (String s: sharedTodo) {
				VariableDeclaration vd = CetusUtil.getDeclarationStatement(proc.getProcedure(), s);
				if (vd==null) continue;
				ArraySpecifier spe = (ArraySpecifier)vd.getDeclarator(0).getArraySpecifiers().get(0);
				List<Expression> exs = new ArrayList();
				exs.add(new IntegerLiteral(mergeAction.getMergeNumber()));
				for (int i=0; i<spe.getNumDimensions(); i++) {
					exs.add((Expression)spe.getDimension(i).clone());
				}
				ArraySpecifier as = new ArraySpecifier(exs);
				vd.getDeclarator(0).getArraySpecifiers().set(0, as);
			}
			
			if (memoryExpressions.size()>0) {
				// ntidx = tidx%blockDim.x
				// ibidx = tidx/blockDim.x;
				Statement ntidx_ds = StatementUtil.loadIntDeclaration(new Identifier("ntidx"));
				Statement nbidx_ds = StatementUtil.loadIntDeclaration(new Identifier("ibidx"));
				AssignmentExpression ntidx_ae = new AssignmentExpression(new Identifier("ntidx"), AssignmentOperator.NORMAL, 
						new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.MODULUS, new IntegerLiteral(proc.getBlockDimX())));
				AssignmentExpression ibidx_ae = new AssignmentExpression(new Identifier("ibidx"), AssignmentOperator.NORMAL, 
						new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.DIVIDE, new IntegerLiteral(proc.getBlockDimX())));
				StatementUtil.addToFirst(proc.getProcedure().getBody(), ntidx_ds);
				StatementUtil.addToFirst(proc.getProcedure().getBody(), nbidx_ds);
				StatementUtil.addSibling(ntidx_ds, new ExpressionStatement(ntidx_ae), false);
				StatementUtil.addSibling(ntidx_ds, new ExpressionStatement(ibidx_ae), false);
				
			}
			
			proc.setBlockDimX(proc.getBlockDimX()*getMergeAction().getMergeNumber());
		}
		else {
			// only the first thread block only load it
			for (BlockDiff blockDiff: getMergeAction().getBlockDiffs()) {
				MemoryExpression memoryExpression = blockDiff.getMemoryExpression();
				GLoop loop = memoryExpression.getLoop();
				boolean unroll = false;
				boolean isIncludedY = false;
				isIncludedY = isIncludedY||CetusUtil.isContain(memoryExpression, new Identifier(ThreadIndex.IDY));
				isIncludedY = isIncludedY||CetusUtil.isContain(memoryExpression, new Identifier(ThreadIndex.TIDY));
				if (loop.getEnd() instanceof IntegerLiteral&&loop.getBody().getChildren().size()==1&&!isIncludedY) {
					int end = (int)((IntegerLiteral)loop.getEnd()).getValue();
					long lsize = (end-loop.getStart())/loop.getIncrement().getValue();
					BinaryExpression newex = null;
					if (getMergeAction().getMergeNumber()==lsize) {
						Expression ex = null;
						ex = new BinaryExpression(new Identifier(ThreadIndex.TIDY), BinaryOperator.MULTIPLY, (IntegerLiteral)loop.getIncrement().clone());
						newex = new BinaryExpression(new IntegerLiteral(loop.getStart()), BinaryOperator.ADD, ex);
						unroll = true;
					}
					else 
					if (lsize%getMergeAction().getMergeNumber()==0){
						Expression ex = null;
						ex = new BinaryExpression(new Identifier(ThreadIndex.TIDY), BinaryOperator.MULTIPLY, (IntegerLiteral)loop.getIncrement().clone());

						newex = new BinaryExpression((Identifier)loop.getIterator().clone(), BinaryOperator.ADD, ex);
						unroll = true;
					}
					
					if (true) {
					
						int ninc = getMergeAction().getMergeNumber()*(int)loop.getIncrement().getValue();
						loop.getIncrement().setValue(ninc);
						CetusUtil.replaceChild(memoryExpression.getStatement(), loop.getIterator(), newex);	
					}
						
				}
				
				if (!unroll) {
					BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.TIDY), 
							BinaryOperator.COMPARE_LT, new IntegerLiteral(1));
					// ifs can be removed if we can map iterator to tidy
					IfStatement ifs = new IfStatement(be, (Statement)memoryExpression.getStatement().clone());
					ifs.swapWith(memoryExpression.getStatement());
				}
			}
			
			// coalesced_idy->(idy-tidy) if (MergeNumber==coalesceNumber)
			if (CudaConfig.getCoalescedThread()%getMergeAction().getMergeNumber()==0) {
//				int v = CudaConfig.getCoalescedThread()/getMergeAction().getMergeNumber();
				if (getMergeAction().getMergeNumber()==CudaConfig.getCoalescedThread()) {
					CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.COALESCED_IDY), 
							new BinaryExpression(new Identifier(ThreadIndex.IDY), BinaryOperator.SUBTRACT, new Identifier(ThreadIndex.TIDY)));
				}
			}
			proc.setBlockDimY(proc.getBlockDimY()*getMergeAction().getMergeNumber());
		}
			
	}

}
