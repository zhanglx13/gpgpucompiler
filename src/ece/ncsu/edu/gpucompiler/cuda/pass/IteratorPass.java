package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;
import cetus.hir.VariableDeclaration;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedureUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;
import ece.ncsu.edu.gpucompiler.cuda.index.ConstIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;
import ece.ncsu.edu.gpucompiler.cuda.index.LoopIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.merger.BlockDiff;
import ece.ncsu.edu.gpucompiler.cuda.merger.MergeAction;
import ece.ncsu.edu.gpucompiler.cuda.merger.Merger;

public class IteratorPass extends MergePass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}


	public int computeDiff(MemoryArrayAccess ma, boolean isX, List<String> ids, List<Integer> diffs) {
		int diff = 0;
		Address address = null;
		if (isX) address = ma.getX();
		else address = ma.getY();
		for (Index in: address.getIndexs()) {
			if (in instanceof ThreadIndex) {
				ThreadIndex in0 = (ThreadIndex)in;
				for (int i=0; i<ids.size(); i++) {
					if (in0.getId().equals(ids.get(i))) {
						diff += in0.getCoefficient()*diffs.get(i);
					}
				}
			}
			else
			if (in instanceof LoopIndex) {
				LoopIndex in0 = (LoopIndex)in;
				for (int i=0; i<ids.size(); i++) {
					if (in0.getId().toString().equals(ids.get(i))) {
						diff += in0.getCoefficient()*diffs.get(i);
					}
				}
			}				
			else 
			if (in instanceof ConstIndex) {
//				ConstIndex in0 = (ConstIndex)in;
//				diff += 0;
			}			
		}
		return diff;
	}		
	
	void computeActionMergeNumber(GProcedure proc, MergeAction action) {
		int reqBlocksize = proc.getBlockDimX()*proc.getBlockDimY();
		int reg = CudaConfig.getDefault().getRegisterInMP();
		int smem = CudaConfig.getDefault().getShareMemoryInMP();
		int blocksize = CudaConfig.getDefault().getThreadInBlock();		
		int reqReg = proc.getRegCount();
		int reqSmem = proc.getSharedMemorySize();
		action.setMergeNumber(1);
		int numY = 1;
		int numX = 1;
		while (true) {
			numY *= 2;
			numX *= 2;
			reqReg *= 4;
			reqSmem *= 2;
			if (action.getMergeType()==Merger.MERGE_TYPE_THREADBLOCK) reqBlocksize=reqBlocksize*2;
			
			if (reg<reqReg) break;
			if (smem<reqSmem) break;
			if (blocksize<reqBlocksize) break;
			
			action.setMergeNumber(numY);
		}		
	}
	
	@Override
	public void dopass(GProcedure func) throws UnsupportedCodeException {
		// we only support ty==1 now
		BreadthFirstIterator bfi = new BreadthFirstIterator(func.getProcedure());

		List<MemoryExpression> memoryExpressions = bfi.getList(MemoryExpression.class);
		List<BlockDiff> yDiffs = new ArrayList();
		for (MemoryExpression memoryExpression: memoryExpressions) {
//			System.out.println("memory: "+memoryExpression);

			GLoop loop = memoryExpression.getLoop();
			List<String> idxs = new ArrayList();
			List<Integer> diffxs = new ArrayList();
			idxs.add(ThreadIndex.IDX);
			diffxs.add(func.getBlockDimX());
			idxs.add(loop.getIterator().toString());
			diffxs.add((int)loop.getIncrement().getValue());

			List<String> idys = new ArrayList();
			List<Integer> diffys = new ArrayList();
			idys.add(ThreadIndex.IDY);
			diffys.add(func.getBlockDimY());
			idys.add(loop.getIterator().toString());
			diffys.add((int)loop.getIncrement().getValue());

			
			// we only consider block0's i equals block1's i+1
			// not consider block0's i+1 equals block1's i
			MemoryArrayAccess ma = memoryExpression.getGlobalMemoryArrayAccess();
			if (ma==null||ma==memoryExpression.getlMemoryArrayAccess()) continue;
			
//			if (ma.getX().isContain(new LoopIndex(loop.getIterator())).size()==0) continue;

			{
				// try to find X neighbor diff
				int xdiff = computeDiff(ma, true, idxs, diffxs);
				int ydiff = ma.getNumIndices()>1?computeDiff(ma, false, idxs, diffxs):0;
				BlockDiff diff = new BlockDiff();
				diff.setMemoryExpression(memoryExpression);
				log(Level.INFO, diff.getMemoryExpression()+":"+xdiff+";"+ydiff);					
				if (xdiff==0&&ydiff==0) {
					// TODO
					log(Level.INFO, diff.getMemoryExpression()+" x iterator pass");					
					diff.setX(0);
				}
			}
			
			{
				// try to find Y neighbor diff
				int xdiff = computeDiff(ma, true, idys, diffys);
				int ydiff = ma.getNumIndices()>1?computeDiff(ma, false, idys, diffys):0;
				BlockDiff diff = new BlockDiff();
				diff.setMemoryExpression(memoryExpression);
				log(Level.INFO, diff.getMemoryExpression()+":"+xdiff+";"+ydiff);					
				if (xdiff==0&&ydiff==0) {
					diff.setY(0);
					if (yDiffs.size()>0) {
						BlockDiff dif = yDiffs.get(0);
						if (dif.getMemoryExpression().getGlobalMemoryArrayAccess().getArrayName().toString().equals(ma.getArrayName().toString())) {
							log(Level.INFO, diff.getMemoryExpression()+" y iterator pass");					
							yDiffs.add(diff);
						}
					}
					else {
						log(Level.INFO, diff.getMemoryExpression()+" y iterator pass");					
						yDiffs.add(diff);
					}
				}
			}
		}
		if (yDiffs.size()>0) {
			MergeAction action = new MergeAction();
			action.setMergeDirection(Merger.MERGE_DIRECTION_Y);
			action.setBlockDiffs(yDiffs);
			action.setMergeType(Merger.MERGE_TYPE_THREAD);
			computeActionMergeNumber(func, action);
			if (action.getMergeNumber()>=4) action.setMergeNumber(action.getMergeNumber()/2);
			if (action.getMergeNumber()>=16) action.setMergeNumber(8);			
			Snapshot ss = new Snapshot();
			ss.setProcedure(func.generateRunnableCode());
			String optimal = null;
			String original = func.generateRunnableCode();
			for (int i=1; i<8; i*=2) {
				System.out.println("start "+i);
				func.refresh(ss.getProcedure());
				reloadMemoryExpression(func, yDiffs);
				try {
					int myi = func.getDefInt(GProcedure.DEF_merger_y);
					func.setDef(GProcedure.DEF_merger_y, myi*action.getMergeNumber()+"");
					
					yfilter(action, func);
					String nid = action.toString();
					func.gerenateOutput(func.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
					func.refresh();
					if (i==2) {
						optimal = func.generateRunnableCode();
					}
				}
				catch (UnsupportedCodeException ex) {
					ex.printStackTrace();
				}		
				action.setMergeNumber(action.getMergeNumber()*2);
			}
			if (optimal!=null) {
				func.refresh(optimal);
			}
			else {
				func.refresh(original);
			}			
			
		}

			/*
		if (yDiffs.size()!=0) {
			// we only keep one if it global to register, if it global to share, we keep list of same share
			BlockDiff diff = yDiffs.get(0);
			MemoryArrayAccess shared = diff.getMemoryExpression().getSharedMemoryArrayAccess();
			
			for (int i=1; i<yDiffs.size(); i++) {
				BlockDiff diff0 = yDiffs.get(i);
				if (share!=null&&share.equals(diff0.getMemoryAccess().getMemoryStatement().getShareMemory())) {
					continue;
				}
				yDiffs.remove(diff0);
				i--;
			}
			filter(ylist, false, maxnumber);
		}
		*/
	}
	
	public void yfilter(MergeAction action, GProcedure proc) {
		// we only consider block0's i equals block1's i+1
		GLoop loop = action.getBlockDiffs().get(0).getMemoryExpression().getLoop();
		int maxnumber = action.getMergeNumber();
		
		int end = Integer.MAX_VALUE/2;
		if (loop.getEnd() instanceof IntegerLiteral) {
			IntegerLiteral il = (IntegerLiteral)loop.getEnd();
			end = (int)il.getValue();
		}
		List<MemoryExpression> mas = new ArrayList();
		List<String> shareds = new ArrayList();
		for (BlockDiff diff: action.getBlockDiffs()) {
			mas.add(diff.getMemoryExpression());
			if (diff.getMemoryExpression().getSharedMemoryArrayAccess()!=null) {
				shareds.add(diff.getMemoryExpression().getSharedMemoryArrayAccess().getArrayName().toString());
			}
			Statement stmt = diff.getMemoryExpression().getStatement();
			Identifier id = new Identifier(ThreadIndex.IDY);
			CetusUtil.replaceChild(stmt, id, new BinaryExpression(id, BinaryOperator.MULTIPLY, new IntegerLiteral(maxnumber)));
		}

		// for sum0, idy*maxnumber, the loop is from start to end
		// for sum1, idy*maxnumber+1, the loop is from start to end, start+1 share data with sum0's start

		List<VariableDeclaration> decsTodo = new ArrayList();
		List<ExpressionStatement> essTodo = new ArrayList();
		
		GProcedureUtil.loadTodos(proc, mas, decsTodo, essTodo);

		for (int i=0; i<essTodo.size(); i++) {
			ExpressionStatement es = essTodo.get(i);
			Expression ex = es.getExpression();
			if (ex instanceof MemoryExpression) {
//				System.out.println("find one share access: "+ ex);
				MemoryExpression memoryExpression = ((MemoryExpression)ex);
				MemoryArrayAccess ma = memoryExpression.getSharedMemoryArrayAccess();
				if (ma==null) continue;
				if (ma!=memoryExpression.getrMemoryArrayAccess()) continue;
				if (shareds.contains(ma.getArrayName().toString())) {
					essTodo.remove(es);
					i--;
				}
			}

		}
		
		
		GProcedureUtil.duplicateDeclaration(proc, decsTodo, maxnumber);
		List<Expression> exlist = new ArrayList();
		for (int i=0; i<maxnumber; i++) {
			exlist.add(new BinaryExpression(loop.getIterator(), BinaryOperator.ADD, new IntegerLiteral(i)));
		}
//		loopStmt.getLoop();
		
		
		
		ForLoop nloop = loop;
		nloop = (ForLoop)nloop.clone();
		StatementUtil.addSibling(loop, nloop, false);
//		System.out.println("s"+exlist.size());
		GProcedureUtil.duplicateStatement(proc, decsTodo, essTodo, 0, maxnumber, false, loop.getIterator(), exlist);
		
		if (end<maxnumber) {
			loop.getParent().removeChild(loop);
		}
		
		
		loop.setCondition(new BinaryExpression(loop.getIterator(), BinaryOperator.COMPARE_LT, 
				new BinaryExpression(loop.getEnd(), BinaryOperator.SUBTRACT, new IntegerLiteral(maxnumber-1))
				));
		
//		for (ExpressionStatement es0: essTodo) {
//			System.out.println("here: "+es0);
//			
//		}
		for (int k=1; k<maxnumber&&k<end+1; k++) {
			DepthFirstIterator dfi = new DepthFirstIterator(nloop);
			List<ExpressionStatement> ess = (List<ExpressionStatement>)dfi.getList(ExpressionStatement.class);
			
			List<ExpressionStatement> nessTodo = new ArrayList();
			for (int i=0; i<ess.size(); i++) {
				ExpressionStatement es = ess.get(i);
//				Traversable tr = es;
				for (ExpressionStatement es0: essTodo) {
					if (es0.toString().equals(es.toString())) {
						nessTodo.add(es);
//						while ((tr=tr.getParent())!=null) {
//							if (tr.equals(nloop)) {
//								nessTodo.add(es);
//							}
//						}					
					}
				}
			}
			List<Expression> negativeExlist = new ArrayList();
			for (int i=0; i<k; i++) {
				negativeExlist.add(0, new BinaryExpression(loop.getEnd(), BinaryOperator.SUBTRACT, new IntegerLiteral(i+1)));
			}
			
			ForLoop oldloop = nloop;
			nloop = (ForLoop)nloop.clone();
			StatementUtil.addSibling(oldloop, nloop, false);
			GProcedureUtil.duplicateStatement(proc, decsTodo, nessTodo, 0, k, false, loop.getIterator(), negativeExlist);
			CetusUtil.replaceChild(oldloop, loop.getIterator(), negativeExlist.get(0));
			StatementUtil.addSibling(oldloop, (Statement)oldloop.getBody().clone(), false);
			proc.getProcedure().getBody().removeChild(oldloop);
		}
		
		
		
		for (int k=1; k<maxnumber; k++) {
			DepthFirstIterator dfi = new DepthFirstIterator(nloop);
			List<ExpressionStatement> ess = (List<ExpressionStatement>)dfi.getList(ExpressionStatement.class);
			
			List<ExpressionStatement> nessTodo = new ArrayList();
			for (int i=0; i<ess.size(); i++) {
				ExpressionStatement es = ess.get(i);
//				System.out.println("find: "+es);
//				Traversable tr = es;
				for (ExpressionStatement es0: essTodo) {
					if (es0.toString().equals(es.toString())) {
						nessTodo.add(es);
//						while ((tr=tr.getParent())!=null) {
//							if (tr.equals(nloop)) {
//								nessTodo.add(es);
//							}
//						}					
					}
				}
			}
//			for (ExpressionStatement es0: nessTodo) {
//				System.out.println("find: "+es0);
//			}
			List<Expression> positiveExlist = new ArrayList();
			for (int i=0; i<maxnumber; i++) {
				if (i-k>=0)
					positiveExlist.add(new IntegerLiteral(i-k));
				else 
					positiveExlist.add(new BinaryExpression(new IntegerLiteral(0), BinaryOperator.SUBTRACT, new IntegerLiteral((k-i))));
			}

			ForLoop oldloop = nloop;
//			System.out.println("oldloop");
//			System.out.println(oldloop);
			nloop = (ForLoop)nloop.clone();
			if (k!=maxnumber-1) StatementUtil.addSibling(oldloop, nloop, false);
			GProcedureUtil.duplicateStatement(proc, decsTodo, nessTodo, k, (end+k)>maxnumber?maxnumber:k+end, false, loop.getIterator(), positiveExlist);
			CetusUtil.replaceChild(oldloop, loop.getIterator(), positiveExlist.get(0));
			StatementUtil.addSibling(oldloop, (Statement)oldloop.getBody().clone(), false);
			proc.getProcedure().getBody().removeChild(oldloop);
//			System.out.println("oldloop--");
//			System.out.println(oldloop);
			
		}
		

		/*
		for (int i=maxnumber-1; i>-1; i--) {
			ForLoop nloop = (ForLoop)loop.clone();
			BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.IDY), BinaryOperator.MULTIPLY, new IntegerLiteral(maxnumber));
			nloop = (ForLoop)replaceChild(nloop, new Identifier(ThreadIndex.IDY), new BinaryExpression(be, BinaryOperator.ADD, new IntegerLiteral(i)));
			nloop = (ForLoop)replaceChild(nloop, loopStmt.getIterator(), new BinaryExpression(loopStmt.getIterator(), BinaryOperator.ADD, new IntegerLiteral(i)));
			addSibling(loop, nloop, false);
		}
		*/
		
		
		try {
			GProcedureUtil.forwardMemoryWrite(proc);
		} catch (UnsupportedCodeException e) {
			e.printStackTrace();
		}
//		System.out.println(proc.getProcedure());
	}
	
	boolean hanleMerge(MergeAction action, GProcedure proc) {
		log(Level.INFO, " apply " + action.toString());	
		if (action.getMergeDirection()==Merger.MERGE_DIRECTION_X) {
//			xfilter(groupDiffs, maxnumber);
			return false;
		}
		else {
			int myi = proc.getDefInt(GProcedure.DEF_merger_y);
			if (myi*action.getMergeNumber()>CudaConfig.getCoalescedThread()) return false;
			proc.setDef(GProcedure.DEF_merger_y, myi*action.getMergeNumber()+"");
			yfilter(action, proc);			
			return true;
		}
	}
	
}
