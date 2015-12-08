package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.List;
import java.util.logging.Level;

import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.Tools;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;
import ece.ncsu.edu.gpucompiler.cuda.index.LoopIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.UnresolvedIndex;

public class PartitionPass extends Pass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}

	@Override
	public void dopass(GProcedure proc) {
		DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
		List<MemoryExpression> list = dfi.getList(MemoryExpression.class);
		boolean allpass = true;
		List<MemoryExpression> memoryExpressions = new ArrayList();
		for (MemoryExpression memoryExpression: list) {
			if (memoryExpression.getGlobalMemoryArrayAccess()!=null) {
				if (testPartitionConflict(memoryExpression.getGlobalMemoryArrayAccess())) {
					allpass = false;
					memoryExpressions.add(memoryExpression);
					log(Level.INFO, memoryExpression+ ": partition conflict");
				}
			}
		}		
		
		if (allpass) {
			log(Level.INFO, "no partition conflict");
			return;
		}
		
		// we got partition conflict
		// 1. if all the memory accesses have IDY, we can exchange IDY and IDX
		// 2. if IDY is not 1, we can do diagonal 
		// 3. try iteration 
		// TODO: some coefficient also make partition conflict, for example, TIDX*256
		// we don't handle such case now
		
		if (proc.getGlobalDimX()>1&&proc.getGlobalDimY()>1) {
			int countY = 0;
			int conflictCount = 0;
			for (MemoryExpression memoryExpression: list) {
				if (memoryExpression.getGlobalMemoryArrayAccess()!=null) {
					MemoryArrayAccess ma = memoryExpression.getGlobalMemoryArrayAccess();
					if (testPartitionConflict(ma))
						conflictCount++;
					for (Index index: ma.getX().getIndexs()) {
						if (index instanceof ThreadIndex) {
							ThreadIndex ti = (ThreadIndex)index;
							if (ti.getId().equals(ThreadIndex.IDY)||
									ti.getId().equals(ThreadIndex.BIDY)||
									ti.getId().equals(ThreadIndex.COALESCED_IDY)) {
								countY++;
							}
						}
					}		
				}
			}
			if (countY==list.size()) {
				transpose(proc);
				String nid = "transpose";
				proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
				return;
			}
			
			diagonal(proc);
			String nid = "diagonal";
			proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
			return;
		}
		
		List<GLoop> loops = new ArrayList();
		for (MemoryExpression memoryExpression: memoryExpressions) {
			if (loops.contains(memoryExpression.getLoop())) continue;
			loops.add(memoryExpression.getLoop());
		}
		int loopdone = 0;
		if (loops.size()>0) {
			for (GLoop loop: loops) {
				if (loopOffset(loop)) loopdone++;
			}
		}
		String nid = "loopOffset";
		proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
		
	}	



	// test for x direction
	boolean testPartitionConflict(MemoryArrayAccess ma) {
		// y is the same for different group, that means: no tidx, idx, ID16X
//		for (Index index: ma.getY().getIndexs()) {
//			// basically, Y is not important for the partition
//		}
		for (Index index: ma.getX().getIndexs()) {
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
//				if (ti.getId().equals(ThreadIndex.ID16Y)) return false;
				// in general, the idx means threads operate global memory without partition conflict
				// however, if idx*256, something like that, still has partition conflict
				// TODO: idx*256
				if (ti.getId().equals(ThreadIndex.IDX)) return false;
//				if (ti.getId().equals(ThreadIndex.TIDY)) return false;
			}
		}
		return true;
	}
	
	boolean loopOffset(GLoop gloop) {
		// we consider "end" is larger than 
		if (gloop.getEndValue()<CudaConfig.getCoalescedThread()*32) return false;
		
		GProcedure proc = gloop.getGProcedure();
		
		CompoundStatement body = gloop.getCompoundStatementBody();
		Identifier new_it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, proc.getProcedure());
		DeclarationStatement new_it_ds = StatementUtil.loadIntDeclaration(new_it);
		CetusUtil.replaceChild(body, gloop.getIterator(), new_it);
		StatementUtil.addSibling(gloop, new_it_ds, true);
		Tools.addSymbols(proc.getProcedure(), new_it_ds.getDeclaration());
		
		// offset = bidx*32;
		Identifier offset = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
		DeclarationStatement offset_ds = StatementUtil.loadIntDeclaration(offset);
		Tools.addSymbols(proc.getProcedure(), offset_ds.getDeclaration());
		BinaryExpression offsetBE =  new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.MULTIPLY, new IntegerLiteral(CudaConfig.getCoalescedThread()));
		AssignmentExpression offsetAE = new AssignmentExpression((Identifier)offset.clone(), AssignmentOperator.NORMAL, offsetBE);
		ExpressionStatement es = new ExpressionStatement(offsetAE);
		StatementUtil.addSibling(gloop, es, true);
		StatementUtil.addSibling(es, offset_ds, true);

		// newit = (it+offset)%end;
		BinaryExpression binaryExpression =  new BinaryExpression((Identifier)gloop.getIterator().clone(), BinaryOperator.ADD, (Identifier)offset.clone());
		binaryExpression =  new BinaryExpression(binaryExpression, BinaryOperator.MODULUS, (Identifier)gloop.getEnd().clone());
		AssignmentExpression assignmentExpression = new AssignmentExpression((Identifier)new_it.clone(), AssignmentOperator.NORMAL, binaryExpression);
		body.addStatementBefore((Statement)body.getChildren().get(0), new ExpressionStatement(assignmentExpression));
		
		return true;
	}


	/**
	 * 
	{
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid%gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
	} 		
	 */
	void diagonal(GProcedure proc) {
		
		
		Identifier nbidy = new Identifier("nbidy");
		Identifier nbidx = new Identifier("nbidx");

		Identifier tmp = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
		
		//  if (width == height) {
	    //blockIdx_y = blockIdx.x;
	    //blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
		CompoundStatement cs_true = new CompoundStatement();
		AssignmentExpression wh_by = new AssignmentExpression(nbidy, AssignmentOperator.NORMAL, new Identifier(ThreadIndex.BIDX));
		BinaryExpression xy = new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.ADD, new Identifier(ThreadIndex.BIDY));
		AssignmentExpression wh_bx = new AssignmentExpression(nbidx, AssignmentOperator.NORMAL, 
				new BinaryExpression(xy, BinaryOperator.MODULUS, new Identifier(ThreadIndex.GRID_DIM_X)));
		cs_true.addStatement(new ExpressionStatement(wh_by));
		cs_true.addStatement(new ExpressionStatement(wh_bx));
		BinaryExpression condition = new BinaryExpression(new Identifier(ThreadIndex.GRID_DIM_X), BinaryOperator.COMPARE_EQ, new Identifier(ThreadIndex.GRID_DIM_Y));
		
		// replace idx, idy -> nbidx*BLOCK_DIM_X+tidx, nbidy*BLOCK_DIM_Y+tidy
		CompoundStatement cs_false = new CompoundStatement();
		BinaryExpression nidx_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_X), BinaryOperator.MULTIPLY, (Identifier)nbidx.clone());
		nidx_be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.ADD, nidx_be);
		BinaryExpression nidy_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_Y), BinaryOperator.MULTIPLY, (Identifier)nbidy.clone());
		nidy_be = new BinaryExpression(new Identifier(ThreadIndex.TIDY), BinaryOperator.ADD, nidy_be);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDY), nidy_be);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDX), nidx_be);
//		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.COALESCED_IDY), (Identifier)nidx.clone());
		
		
		//int bid = blockIdx.x + gridDim.x*blockIdx.y;
		DeclarationStatement tmp_ds = StatementUtil.loadIntDeclaration(tmp);
		BinaryExpression ae_tmp = new BinaryExpression(new Identifier(ThreadIndex.GRID_DIM_X), BinaryOperator.MULTIPLY, new Identifier(ThreadIndex.BIDY));
		ae_tmp = new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.ADD, ae_tmp);
		ae_tmp = new AssignmentExpression((Identifier)tmp.clone(), AssignmentOperator.NORMAL, ae_tmp);

		//blockIdx_y = bid%gridDim.y;
		BinaryExpression ae0 = new BinaryExpression((Identifier)tmp.clone(), BinaryOperator.MODULUS, new Identifier(ThreadIndex.GRID_DIM_Y));
		ae0 = new AssignmentExpression(nbidy, AssignmentOperator.NORMAL, ae0);
		
		//blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
		BinaryExpression ae1 = new BinaryExpression((Identifier)tmp.clone(), BinaryOperator.DIVIDE, new Identifier(ThreadIndex.GRID_DIM_Y));
		ae1 = new BinaryExpression(ae1, BinaryOperator.ADD, nbidy);
		ae1 = new BinaryExpression(ae1, BinaryOperator.MODULUS, new Identifier(ThreadIndex.GRID_DIM_X));
		ae1 = new AssignmentExpression(nbidx, AssignmentOperator.NORMAL, ae1);
		
		cs_false.addStatement(new ExpressionStatement(ae_tmp));
		cs_false.addStatement(new ExpressionStatement(ae0));
		cs_false.addStatement(new ExpressionStatement(ae1));
		IfStatement ifs = new IfStatement(condition, cs_true, cs_false);
		
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.BIDY), nbidy);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.BIDX), nbidx);
		
		
		
		Statement first = StatementUtil.getFirstDeclarationStatement(proc.getProcedure().getBody());
		DeclarationStatement nbidy_ds = StatementUtil.loadIntDeclaration(nbidy);
		DeclarationStatement nbidx_ds = StatementUtil.loadIntDeclaration(nbidx);
		StatementUtil.addSibling(first, tmp_ds, true);
		StatementUtil.addSibling(first, nbidy_ds, true);
		StatementUtil.addSibling(first, nbidx_ds, true);

		
		StatementUtil.addSibling(first, ifs, true);
		Tools.addSymbols(proc.getProcedure(), tmp_ds.getDeclaration());
		proc.setDef("coalesced_idy", "(nbidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)");
		
	}
	
	void transpose(GProcedure proc) {
		Identifier nbidy = new Identifier("nbidy");
		Identifier nbidx = new Identifier("nbidx");

		// replace idx, idy -> nbidx*BLOCK_DIM_X+tidx, nbidy*BLOCK_DIM_Y+tidy
		BinaryExpression nidx_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_X), BinaryOperator.MULTIPLY, (Identifier)nbidx.clone());
		nidx_be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.ADD, nidx_be);
		BinaryExpression nidy_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_Y), BinaryOperator.MULTIPLY, (Identifier)nbidy.clone());
		nidy_be = new BinaryExpression(new Identifier(ThreadIndex.TIDY), BinaryOperator.ADD, nidy_be);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDY), nidy_be);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.IDX), nidx_be);
//		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.COALESCED_IDY), (Identifier)nidx.clone());
		
		
		//int bid = blockIdx.x + gridDim.x*blockIdx.y;
		
		AssignmentExpression ae0 = new AssignmentExpression((Identifier)nbidy.clone(), AssignmentOperator.NORMAL, new Identifier(ThreadIndex.BIDX));
		AssignmentExpression ae1 = new AssignmentExpression((Identifier)nbidx.clone(), AssignmentOperator.NORMAL, new Identifier(ThreadIndex.BIDY));

		
		
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.BIDY), nbidy);
		CetusUtil.replaceChild(proc.getProcedure().getBody(), new Identifier(ThreadIndex.BIDX), nbidx);
		
		Statement first = StatementUtil.getFirstDeclarationStatement(proc.getProcedure().getBody());
		DeclarationStatement nidy_ds = StatementUtil.loadIntDeclaration(nbidy);
		DeclarationStatement nidx_ds = StatementUtil.loadIntDeclaration(nbidx);
		StatementUtil.addSibling(first, nidx_ds, true);
		StatementUtil.addSibling(first, nidy_ds, true);
		StatementUtil.addSibling(first, new ExpressionStatement(ae0), true);
		StatementUtil.addSibling(first, new ExpressionStatement(ae1), true);
		Tools.addSymbols(proc.getProcedure(), nidx_ds.getDeclaration());		
		Tools.addSymbols(proc.getProcedure(), nidy_ds.getDeclaration());		
	}
	
	public void filter(List<MemoryArrayAccess> mas, GProcedure func) {
		
		
		boolean allpass = true;
		for (MemoryArrayAccess ma: mas) {
			if (testPartitionConflict(ma)) {
				allpass = false;
			}
		}
		
		if (allpass) {
			log(Level.INFO, "no partition conflict");
			return;
		}
		
		// we got partition conflict
		// 1. if all the memory accesses have IDY, we can exchange IDY and IDX
		// 2. if IDY is not 1, we can do diagonal 
		// 3. try iteration 
		// TODO: some coefficient also make partition conflict, for example, TIDX*256
		// we don't handle such case now
		
		int countY = 0;
		int conflictCount = 0;
		for (MemoryArrayAccess ma: mas) {
			if (testPartitionConflict(ma))
				conflictCount++;
			for (Index index: ma.getX().getIndexs()) {
				if (index instanceof ThreadIndex) {
					ThreadIndex ti = (ThreadIndex)index;
					if (ti.getId().equals(ThreadIndex.IDY)||
							ti.getId().equals(ThreadIndex.BIDY)||
							ti.getId().equals(ThreadIndex.COALESCED_IDY)) {
						countY++;
					}
				}
			}		
		}
		Identifier nbidy = new Identifier("nbidy");
		Identifier nbidx = new Identifier("nbidx");
		Identifier nidy = new Identifier("nidy");
		Identifier nidx = new Identifier("nidx");
		Identifier bid = new Identifier("bid");
		CetusUtil.replaceChild(func.getProcedure().getBody(), new Identifier(ThreadIndex.BIDY), (Identifier)nbidy.clone());
		CetusUtil.replaceChild(func.getProcedure().getBody(), new Identifier(ThreadIndex.BIDX), (Identifier)nbidx.clone());
		CetusUtil.replaceChild(func.getProcedure().getBody(), new Identifier(ThreadIndex.IDY), (Identifier)nidy.clone());
		CetusUtil.replaceChild(func.getProcedure().getBody(), new Identifier(ThreadIndex.IDX), (Identifier)nidx.clone());
		Statement previous = null;

		if (countY==mas.size()&&func.getGlobalDimX()>1&&func.getGlobalDimY()==func.getGlobalDimX()) {
	
			
			AssignmentExpression ae0 = new AssignmentExpression((Identifier)nbidy.clone(), AssignmentOperator.NORMAL, new Identifier(ThreadIndex.BIDX));
			AssignmentExpression ae1 = new AssignmentExpression((Identifier)nbidx.clone(), AssignmentOperator.NORMAL, new Identifier(ThreadIndex.BIDY));
			StatementUtil.addToFirst(func.getProcedure().getBody(), new ExpressionStatement(ae1));
			StatementUtil.addToFirst(func.getProcedure().getBody(), new ExpressionStatement(ae0));
			
			// we can do block (x,y) to (y,(x+y)%x)
		}
		else 
		if (countY==0) {
			// without y in any x address
			// try different iteration or other ways
			//System.out.println("countY=0");
			Hashtable<String, GLoop> loops = new Hashtable<String, GLoop>();
			for (MemoryArrayAccess ma: mas) {
				GLoop loopS = ma.getMemoryExpression().getLoop();
				while (loopS!=null) {
					loops.put(loopS.getIterator().getName(), loopS);
					loopS = loopS.getLoopParent();
//					System.out.println(loopS);
				}
				
			}
			Enumeration<String> ks = loops.keys();
			String id = null;
			int max = 0;
			while (ks.hasMoreElements()) {
				String k = ks.nextElement();
				int count = 0;
				for (MemoryArrayAccess ma: mas) {
//					System.out.println(ma.getArrayAccess().getArrayName()+":"+ma.getLoop().getIterator());
					if (testPartitionConflict(ma)) {
						for (Index index: ma.getX().getIndexs()) {
							if (index instanceof LoopIndex) {
								LoopIndex ti = (LoopIndex)index;
								if (ti.getId().getName().equals(k)) {
									count++;
								}
							}
							else
							if (index instanceof UnresolvedIndex) {
								UnresolvedIndex ti = (UnresolvedIndex)index;
								if (ti.getId().getName().equals(k)) {
									count++;
								}
							}
						}		
						
					}
				}				
				if (count>max) {
					max = count;
					id = k;
				}
			}
			//System.out.println("update iterator:"+id+" for "+max);
			GLoop loopStmt = loops.get(id);
			ForLoop loop = loopStmt;
//			init, end, step
//			init+offset, end+offset
//			init+(i-init-offset)%(end-init)

			String offset = "start"+loopStmt.getIterator().toString();
			Statement start = StatementUtil.loadIntDeclaration(new Identifier(offset));
			BinaryExpression be0 = new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.MULTIPLY, loopStmt.getIncrement());
			AssignmentExpression ae0 = new AssignmentExpression(new Identifier(offset), AssignmentOperator.NORMAL, be0);
			be0 = new BinaryExpression(new Identifier(offset), BinaryOperator.ADD, loopStmt.getStartIL());
			StatementUtil.addSibling(loopStmt, start, true);
			StatementUtil.addSibling(loopStmt, new ExpressionStatement(ae0), true);
			
			AssignmentExpression ae1 = new AssignmentExpression(loopStmt.getIterator(), AssignmentOperator.NORMAL, be0);
			loop.setInitialStatement(new ExpressionStatement(ae1));
			be0 = new BinaryExpression(new Identifier(offset), BinaryOperator.ADD, loopStmt.getEnd());
			loop.setCondition(loopStmt.setEndValue(be0));
			loop.getStep();

			String ni = "new"+loopStmt.getIterator().toString();
			start = StatementUtil.loadIntDeclaration(new Identifier(ni));
			
			be0 =  new BinaryExpression(loopStmt.getIterator(), BinaryOperator.SUBTRACT, loopStmt.getStartIL());
			be0 =  new BinaryExpression(be0, BinaryOperator.SUBTRACT, new Identifier(offset));
			BinaryExpression be1 = new BinaryExpression(loopStmt.getEnd(), BinaryOperator.SUBTRACT, loopStmt.getStartIL());
			be0 =  new BinaryExpression(be0, BinaryOperator.MODULUS, be1);
			be0 =  new BinaryExpression(be0, BinaryOperator.ADD, loopStmt.getStartIL());
			ae0 = new AssignmentExpression(new Identifier(ni), AssignmentOperator.NORMAL, be0);
			CetusUtil.replaceChild(loopStmt.getBody(), loopStmt.getIterator(), new Identifier(ni));
			StatementUtil.addToFirst((CompoundStatement)loopStmt.getBody(), new ExpressionStatement(ae0));
			StatementUtil.addToFirst((CompoundStatement)loopStmt.getBody(), start);
			
			
			
			
		}
		else {
			// the indexX include Y, therefore, if exchange the x and y of block.
			// then indexX include x, which means it without partition conflict
			
			// bidy, idy, tidy, coalesced_idy and bidx, idx, tidx, coalesced_idx
			
//			  if (width == height) {
//				    blockIdx_y = blockIdx.x;
//				    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
//			  } else {
//			    int bid = blockIdx.x + gridDim.x*blockIdx.y;
//			    blockIdx_y = bid%gridDim.y;
//			    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
//			  } 		
			BinaryExpression ae_bid = new BinaryExpression(new Identifier(ThreadIndex.GRID_DIM_X), BinaryOperator.MULTIPLY, new Identifier(ThreadIndex.BIDY));
			ae_bid = new BinaryExpression(new Identifier(ThreadIndex.BIDX), BinaryOperator.ADD, ae_bid);
			ae_bid = new AssignmentExpression(bid, AssignmentOperator.NORMAL, ae_bid);
			BinaryExpression ae0 = new BinaryExpression(bid, BinaryOperator.MODULUS, new Identifier(ThreadIndex.GRID_DIM_Y));
			ae0 = new AssignmentExpression(nbidy, AssignmentOperator.NORMAL, ae0);
			
			BinaryExpression ae1 = new BinaryExpression(bid, BinaryOperator.DIVIDE, new Identifier(ThreadIndex.GRID_DIM_Y));
			ae1 = new BinaryExpression(ae1, BinaryOperator.ADD, nbidy);
			
			ae1 = new BinaryExpression(ae1, BinaryOperator.MODULUS, new Identifier(ThreadIndex.GRID_DIM_X));
			ae1 = new AssignmentExpression(nbidx, AssignmentOperator.NORMAL, ae1);
			previous = new ExpressionStatement(ae1);
			
			StatementUtil.addToFirst(func.getProcedure().getBody(), previous);
			StatementUtil.addToFirst(func.getProcedure().getBody(), new ExpressionStatement(ae0));			
			StatementUtil.addToFirst(func.getProcedure().getBody(), new ExpressionStatement(ae_bid));
		}
		
		BinaryExpression nidx_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_X), BinaryOperator.MULTIPLY, nbidx);
		nidx_be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.ADD, nidx_be);
		nidx_be = new BinaryExpression(nidx, AssignmentOperator.NORMAL, nidx_be);
		
		BinaryExpression nidy_be = new BinaryExpression(new Identifier(ThreadIndex.BLOCK_DIM_Y), BinaryOperator.MULTIPLY, nbidy);
		nidy_be = new BinaryExpression(new Identifier(ThreadIndex.TIDY), BinaryOperator.ADD, nidy_be);
		nidy_be = new BinaryExpression(nidy, AssignmentOperator.NORMAL, nidy_be);
		
		StatementUtil.addSibling(previous, new ExpressionStatement(nidx_be), false);
		StatementUtil.addSibling(previous, new ExpressionStatement(nidy_be), false);
		
		StatementUtil.addToFirst(func.getProcedure().getBody(), StatementUtil.loadIntDeclaration((Identifier)nbidy.clone()));
		StatementUtil.addToFirst(func.getProcedure().getBody(), StatementUtil.loadIntDeclaration((Identifier)nbidy.clone()));
		StatementUtil.addToFirst(func.getProcedure().getBody(), StatementUtil.loadIntDeclaration((Identifier)nbidx.clone()));
		StatementUtil.addToFirst(func.getProcedure().getBody(), StatementUtil.loadIntDeclaration((Identifier)bid.clone()));
		
		//System.out.println(func.getProcedure());
	}

}
