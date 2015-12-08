package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import cetus.hir.DepthFirstIterator;
import cetus.hir.Procedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;
import ece.ncsu.edu.gpucompiler.cuda.index.ConstIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.merger.BlockDiff;
import ece.ncsu.edu.gpucompiler.cuda.merger.MergeAction;
import ece.ncsu.edu.gpucompiler.cuda.merger.Merger;
import ece.ncsu.edu.gpucompiler.cuda.merger.ThreadBlockMerger;
import ece.ncsu.edu.gpucompiler.cuda.merger.ThreadMerger;

public class MergePass extends Pass {
	
	int actionNumber = 0;

	
	// test for x direction
	boolean testXGroup(MemoryArrayAccess ma) {

		
		// y is the same for different group in x direction, that means: no tidx, idx, COALESCED_IDX
		if (ma.getY()!=null)
		for (Index index: ma.getY().getIndexs()) {
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.COALESCED_IDX)) return false;
				if (ti.getId().equals(ThreadIndex.IDX)) return false;
				if (ti.getId().equals(ThreadIndex.TIDX)) return false;
			}
		}
		return true;
	}
	
	// y is the same for different group in y direction, that means: no tidy, idy, COALESCED_IDY
	boolean testYGroup(MemoryArrayAccess ma) {
		if (ma.getY()!=null)
		for (Index index: ma.getY().getIndexs()) {
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
//				if (ti.getId().equals(ThreadIndex.COALESCED_IDY)) return false;
				if (ti.getId().equals(ThreadIndex.IDY)) return false;
//				if (ti.getId().equals(ThreadIndex.TIDY)) return false;
			}
		}
		return true;
	}
			

	
	public BlockDiff computeDiff(MemoryArrayAccess ma) {
		int coalescedNumber = CudaConfig.getCoalescedThread();		
		BlockDiff diff = new BlockDiff();
		if (testXGroup(ma)) 
		{
			// 1. compute the different in X direction
			int offset = 0;
			for (Index index: ma.getX().getIndexs()) {
				if (index instanceof ThreadIndex) {
					// neighbor offset is coalescedNumber
					ThreadIndex ind = (ThreadIndex)index;
					if (ind.getId().equals(ThreadIndex.COALESCED_IDX)) offset+=coalescedNumber*ind.getCoefficient();
					if (ind.getId().equals(ThreadIndex.IDX)) offset+=coalescedNumber*ind.getCoefficient();					
				}
			}
			diff.setX(offset);
		}

		
		if (testYGroup(ma)) 
		{
			// 2. compute the different in Y direction
			// ok, try to put neighbor groups into together 
			// we compute the offset of two neighbor groups for this memory access
			int offset = 0;
			for (Index index: ma.getX().getIndexs()) {
				if (index instanceof ThreadIndex) {
					ThreadIndex ind = (ThreadIndex)index;
					// TODO: we only merge at most coalescedNumber in Y direction
					// so that the offset for is 0
					if (ind.getId().equals(ThreadIndex.COALESCED_IDY)) offset+=0;	
					if (ind.getId().equals(ThreadIndex.IDY)) offset+=1;					
				}
			}
			diff.setY(offset);
		}
		diff.setMemoryExpression(ma.getMemoryExpression());

		
		return diff;
		
		
	}
	
	

	
	public BlockDiff computeOverlapDiff(MemoryArrayAccess ma, MemoryArrayAccess mb) {
		if (ma.getNumIndices()!=mb.getNumIndices()) return null;
		if (!ma.getArrayName().toString().equals(mb.getArrayName().toString())) return null;
		Address ay = ma.getY();
		Address ax = ma.getX();
		Address by = mb.getY();
		Address bx = mb.getX();
		if (ay!=null) {

			Address addr = ay.subtract(by);
			if (addr.getIndexs().size()==0) {
			}
			else
			if (addr.getIndexs().size()==1) {
				Index in = addr.getIndexs().get(0);
				if (!(in instanceof ConstIndex)) return null;
				ConstIndex ci = (ConstIndex)in;
				if (ci.getCoefficient()!=0) return null;
			}
			else {
				return null;
			}
		}
		Address newAddress = new Address();
		
		int tx = ma.getMemoryExpression().getLoop().getGProcedure().getBlockDimX();
		
		int offseta = 0;
		for (Index index: ax.getIndexs()) {
			Index in = (Index)index.clone();
			if (in instanceof ThreadIndex) {
				if (((ThreadIndex) in).getId().equals(ThreadIndex.IDX)) {
					offseta += in.getCoefficient();
					continue;
				}
			}
			in.setCoefficient(in.getCoefficient());
			newAddress.getIndexs().add(in);
		}
		int offsetb = 0;
		for (Index index: bx.getIndexs()) {
			Index in = (Index)index.clone();
			if (in instanceof ThreadIndex) {
				if (((ThreadIndex) in).getId().equals(ThreadIndex.IDX)) {
					offsetb += tx*in.getCoefficient();
					continue;
				}
			}
			in.setCoefficient(-in.getCoefficient());
			newAddress.getIndexs().add(in);
		}
		
		if (offseta*tx!=offsetb) return null;
		newAddress.compact();
		
		if (newAddress.getIndexs().size()==0) {
			newAddress.getIndexs().add(new ConstIndex(0));
		}
		if (newAddress.getIndexs().size()==1) {
			Index in = newAddress.getIndexs().get(0);
			if (!(in instanceof ConstIndex)) return null;
			ConstIndex ci = (ConstIndex)in;
			if (ci.getCoefficient()!=offsetb) return null;
			BlockDiff diff = new BlockDiff();
			diff.setMemoryExpression(mb.getMemoryExpression());
			if (ma.getMemoryExpression()!=mb.getMemoryExpression())
				diff.setMemoryExpressionR(ma.getMemoryExpression());
			diff.setX(0);
			return diff;
		}
		
		return null;
		
	}

	int xNumber = -1;
	int yNumber = -1;
		
	public MergePass() {
	}
	
	
	
	public int getxNumber() {
		return xNumber;
	}

	public void setxNumber(int xNumber) {
		this.xNumber = xNumber;
	}

	public int getyNumber() {
		return yNumber;
	}

	public void setyNumber(int yNumber) {
		this.yNumber = yNumber;
	}

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
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
			if (action.getMergeType()==Merger.MERGE_TYPE_THREADBLOCK) {
				reqBlocksize=reqBlocksize*2;
			}
			
			if (reg<reqReg) break;
			if (smem<reqSmem) break;
			if (blocksize<reqBlocksize) break;
			
			action.setMergeNumber(numY);
		}		
	}

	
	/**
	 * BlockDiff is possible:
	 * 	x = 0;
	 *  y = 0;
	 *  shared memory is or not null: G2S or G2R
	 * therefore
	 * 1. find all x=0
	 * 1.1 if there is one G2S. We do thread block merge
	 * 1.2 if all are G2R, we do thread merge
	 * 2. find all y=0
	 * 2.1 if there is one G2S or we did thread merge in 1.2. We do thread block merge
	 * 2.2 otherwise we do thread merge
	 * @param blockDiffs
	 */
	List<MergeAction> computeMergeAction(GProcedure proc, List<BlockDiff> blockDiffs) {
		int reqBlocksize = proc.getBlockDimX()*proc.getBlockDimY();
//		System.out.println("reqBlocksize="+reqBlocksize);

		List<MergeAction> actions = new ArrayList();
		int reg = CudaConfig.getDefault().getRegisterInMP();
		int smem = CudaConfig.getDefault().getShareMemoryInMP();
		int blocksize = CudaConfig.getDefault().getThreadInBlock();
		
		MergeAction actionX = new MergeAction();
		boolean isTryY = true;
		if (proc.getGlobalDimX()>1) {
			actionX.setMergeDirection(Merger.MERGE_DIRECTION_X);
			actionX.setMergeType(Merger.MERGE_TYPE_THREAD);
			for (BlockDiff diff: blockDiffs) {
				if (diff.getX()==0) {
					actionX.getBlockDiffs().add(diff);
					if (diff.getMemoryExpression().getSharedMemoryArrayAccess()!=null) {
						actionX.setMergeType(Merger.MERGE_TYPE_THREADBLOCK);
					}
					if (diff.getMemoryExpressionR()!=null&&(diff.getMemoryExpressionR()!=diff.getMemoryExpression())) {
						actionX.setMergeType(Merger.MERGE_TYPE_THREADBLOCK);
						// for different statement sharing, we don't do Y direction thread (block) merge
						isTryY = false;
					}
				}
			}
			
			if (actionX.getMergeType()==Merger.MERGE_TYPE_THREADBLOCK) {
				/*
				Procedure procedure = proc.getProcedure();
				DepthFirstIterator dfi = new DepthFirstIterator(procedure);
				List<MemoryExpression> list = dfi.getList(MemoryExpression.class);
				for (BlockDiff bd: actionX.getBlockDiffs()) {
					MemoryExpression me = bd.getMemoryExpression();
					MemoryExpression mer = bd.getMemoryExpressionR();
					for (int i=0; i<list.size(); i++) {
						MemoryExpression memoryExpression = list.get(i);
						if (memoryExpression.getId()==me.getId()) {
							list.remove(memoryExpression);
							i--;
						}
						else
						if (mer!=null&&memoryExpression.getId()==me.getId()) {
							list.remove(memoryExpression);
							i--;
						}
					}
					
				}	
				if (list.size()!=0) {
					for (MemoryExpression memoryExpression: list) {
						MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
						if (maa!=null) {
							if (maa.getX().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) {
								actionX.setMergeType(Merger.MERGE_TYPE_THREAD);
								break;
							}
							if (maa.getY()!=null&&maa.getX().isContain(ThreadIndex.getThreadIndex(ThreadIndex.TIDX)).size()!=0) {
								actionX.setMergeType(Merger.MERGE_TYPE_THREAD);
								break;
							}
						}
					}
				}
				*/
			}
			
		}
		
		
		MergeAction actionY = new MergeAction();
//		System.out.println("isTryY:"+isTryY);
		if (proc.getGlobalDimY()>1&&isTryY) {
			actionY.setMergeDirection(Merger.MERGE_DIRECTION_Y);
			actionY.setMergeType(Merger.MERGE_TYPE_THREAD);
			if (actionX.getBlockDiffs().size()>0&&actionX.getMergeType()==Merger.MERGE_TYPE_THREAD) {
				actionY.setMergeType(Merger.MERGE_TYPE_THREADBLOCK);
			}
			if (actionX.getBlockDiffs().size()==0&&reqBlocksize*4<blocksize) {
				actionY.setMergeType(Merger.MERGE_TYPE_THREADBLOCK);			
			}
			for (BlockDiff diff: blockDiffs) {
				if (diff.getMemoryExpressionR()!=null&&(diff.getMemoryExpressionR()!=diff.getMemoryExpression())) {
					continue;
				}			
				if (diff.getY()==0) {
//					if (!actionX.getBlockDiffs().contains(diff))
						actionY.getBlockDiffs().add(diff);
				}
			}
		}

		

		
		if (actionY.getBlockDiffs().size()>0&&actionX.getBlockDiffs().size()>0) {
			int reqReg = proc.getRegCount();
			int reqSmem = proc.getSharedMemorySize();
			actionY.setMergeNumber(1);
			actionX.setMergeNumber(1);	
			int numY = 1;
			int numX = 1;
			while (true) {
				numY *= 2;
				numX *= 2;
				reqReg *= 4;
				reqSmem *= 2;
				if (actionY.getMergeType()==Merger.MERGE_TYPE_THREADBLOCK) reqBlocksize=reqBlocksize*2;
				if (actionX.getMergeType()==Merger.MERGE_TYPE_THREADBLOCK) reqBlocksize=reqBlocksize*2;
				
				if (reg<reqReg) break;
				if (smem<reqSmem) break;
				if (blocksize<reqBlocksize) break;
				
				actionY.setMergeNumber(numY);
				actionX.setMergeNumber(numX);	
			}
		}
		else 
		if (actionY.getBlockDiffs().size()>0||actionX.getBlockDiffs().size()>0) {
			MergeAction action = actionY.getBlockDiffs().size()>0?actionY:actionX;
			int reqReg = proc.getRegCount();
			float reqSmem = proc.getSharedMemorySize();
			action.setMergeNumber(1);
			int num = 1;
			while (true) {
				num *= 2;
				reqReg *= 2;
				if (action.getMergeType()==Merger.MERGE_TYPE_THREAD) {
					reqSmem *= 2;
					reqReg *= 2;					
				}
				else {
					reqBlocksize=reqBlocksize*2;
				}
				
				if (reg<reqReg) break;
				if (smem<reqSmem) break;
				if (blocksize<reqBlocksize) break;	
				
				action.setMergeNumber(num);
				if (num==CudaConfig.getCoalescedThread()) break;
			}
		}
		
		if (xNumber!=-1) actionX.setMergeNumber(xNumber);
		if (yNumber!=-1) actionY.setMergeNumber(yNumber);
		
		if (actionX.getBlockDiffs().size()>0) {
//			System.out.println(actionX.toString());
			actions.add(actionX);
		}
		if (actionY.getBlockDiffs().size()>0) {
//			System.out.println(actionY.toString());
			actions.add(actionY);
		}
		
		/*
//		System.out.println("------------"+actionX);
		for (BlockDiff diff: actionX.getBlockDiffs()) {
			System.out.println(diff.getMemoryExpression());
		}
		System.out.println(actionY);
		for (BlockDiff diff: actionY.getBlockDiffs()) {
			System.out.println(diff.getMemoryExpression());
		}
		*/
		
		return actions;
	}
	
	
	
	
	public int getActionNumber() {
		return actionNumber;
	}

	public void setActionNumber(int actionNumber) {
		this.actionNumber = actionNumber;
	}

	@Override
	public void dopass(GProcedure proc) throws UnsupportedCodeException {
		//1. load global memory expression
		DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
		List<MemoryExpression> list = dfi.getList(MemoryExpression.class);
		List<BlockDiff> blockDiffs = new ArrayList();

		List<MemoryExpression> memoryExpressions = new ArrayList();
		for (MemoryExpression memoryExpression: list) {
			if (memoryExpression.getGlobalMemoryArrayAccess()!=null&&memoryExpression.getGlobalMemoryArrayAccess()==memoryExpression.getrMemoryArrayAccess()) {
				memoryExpressions.add(memoryExpression);
			}
		}		
		
		for (MemoryExpression memoryExpressionA: memoryExpressions) {
			for (MemoryExpression memoryExpressionB: memoryExpressions) {
				if (memoryExpressionA==memoryExpressionB) {
					// memory statement must be shared[] = global[]
					MemoryArrayAccess maa = memoryExpressionA.getGlobalMemoryArrayAccess();
					//2. find data sharing
					BlockDiff diff = computeDiff(maa);
					if (diff!=null&&(diff.getX()==0||diff.getY()==0)) {
						blockDiffs.add(diff);
						log(Level.INFO, memoryExpressionA+"offset:x="+diff.getX()+";y="+diff.getY());
					}					
				}
				else {
					BlockDiff diff = computeOverlapDiff(memoryExpressionA.getGlobalMemoryArrayAccess(), memoryExpressionB.getGlobalMemoryArrayAccess());
					if (diff!=null&&(diff.getX()==0||diff.getY()==0)) {
						blockDiffs.add(diff);
						log(Level.INFO, memoryExpressionA+" vs "+memoryExpressionB+":offset:x="+diff.getX()+";y="+diff.getY());
					}				
				}
			}
		}		
		
//		for (MemoryExpression memoryExpression: memoryExpressions) {
//
//			if (memoryExpression.getGlobalMemoryArrayAccess()!=null&&memoryExpression.getGlobalMemoryArrayAccess()==memoryExpression.getrMemoryArrayAccess()) {
//				// memory statement must be shared[] = global[]
//				MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
//				//2. find data sharing
//				BlockDiff diff = computeDiff(maa);
//				if (diff!=null&&(diff.getX()==0||diff.getY()==0)) {
//					blockDiffs.add(diff);
//					log(Level.INFO, memoryExpression+"offset:x="+diff.getX()+";y="+diff.getY());
//				}
//			}
//		}		
//		

		Snapshot ss = new Snapshot();
		ss.setProcedure(proc.generateRunnableCode());
		//3. make decision
		//3.1 compute actions
		//3.2 we have at most two action and expected number
		//    we try (num0/2, num0, num0*2) (num1/2, num1, num1*2), in total 9 possible  
		List<MergeAction> actions = computeMergeAction(proc, blockDiffs);
		if (actions.size()==1) {
			actionNumber = 1;
			MergeAction action = actions.get(0);
			if (action.getMergeNumber()==1) return;
//			hanleMerge(action, proc);
//			proc.refresh();
//			String nid = proc.getProcedure().getName().toString()+"_"+action.getMergeNumber();
//			gerenateCopy(getName()+"_"+nid, proc.generateRunnableCode());
//			System.out.println(proc.getProcedure().toString());
			if (action.getMergeNumber()>=2) action.setMergeNumber(action.getMergeNumber()/2);
			String optimal = null;
			String original = proc.generateRunnableCode();
			for (int i=0; i<3; i++) {
				//4. merge and test the kernel is ok
				if (action.getMergeNumber()!=1) {
				
					proc.refresh(ss.getProcedure());
					reloadMemoryExpression(proc, blockDiffs);
					try {
						if (hanleMerge(action, proc)) {
							proc.refresh();				
							proc.testResource();
							String nid = action.toString();
							if (optimal==null) optimal = proc.generateRunnableCode();
							if (i==1) optimal = proc.generateRunnableCode();
							proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
						}	
					}
					catch (UnsupportedCodeException ex) {
//						ex.printStackTrace();
						System.out.println(ex.getMessage());
					}					
				}
				action.setMergeNumber(action.getMergeNumber()*2);
			}
			if (optimal!=null) {
				proc.refresh(optimal);
			}
			else {
				proc.refresh(original);
			}
		}
		else
		if (actions.size()==2) {
			actionNumber = 2;

			MergeAction actionX = actions.get(0);
			MergeAction actionY = actions.get(1);
//			hanleMerge(actionX, proc);
//			System.out.println(proc.getProcedure().toString());
//			proc.refresh();
//			hanleMerge(actionY, proc);
//			proc.refresh();			
//			new NullLoopPass(false).dopass(proc);
//			String nid = proc.getProcedure().getName().toString()+"_"+actionX.getMergeNumber()+"_"+actionY.getMergeNumber();
//			gerenateCopy(getName()+"_"+nid, proc.generateRunnableCode());
//			System.out.println(proc.getProcedure().toString());
			if (actionX.getMergeNumber()>=2) actionX.setMergeNumber(actionX.getMergeNumber()/2);
			if (actionY.getMergeNumber()>=2) actionY.setMergeNumber(actionY.getMergeNumber()/2);
			String optimal = null;
			String original = proc.generateRunnableCode();
			for (int i=0; i<3; i++) {
				for (int j=0; j<3; j++) {
					//4. merge and test the kernel is ok
					proc.refresh(ss.getProcedure());
					
					try {
						reloadMemoryExpression(proc, blockDiffs);
						if (actionX.getMergeNumber()!=1) {
							if (hanleMerge(actionX, proc)) {
								proc.refresh();
								proc.testResource();
							}
						}
						if (actionY.getMergeNumber()!=1) {
							if (hanleMerge(actionY, proc)) {
								proc.refresh();
								proc.testResource();
							}
						}
						String nid = actionX.toString()+"_"+actionY.toString();
						proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+nid);
						if ((i==1&&j==1)||optimal==null) optimal = proc.generateRunnableCode();
					}
					catch (UnsupportedCodeException ex) {
//						ex.printStackTrace();
						System.out.println(ex.getMessage());
					}
	
					actionY.setMergeNumber(actionY.getMergeNumber()*2);
				}
				actionX.setMergeNumber(actionX.getMergeNumber()*2);
				actionY.setMergeNumber(actionY.getMergeNumber()/8);
			}
			if (optimal!=null) {
				proc.refresh(optimal);
			}
			else {
				proc.refresh(original);
			}
			
		}
	}
	
	boolean hanleMerge(MergeAction action, GProcedure proc) {
		log(Level.INFO, " apply " + action.toString());	
		Merger merger = null;
		if (action.getMergeType()==Merger.MERGE_TYPE_THREAD) {
			merger = new ThreadMerger();
			if (action.getMergeDirection()==Merger.MERGE_DIRECTION_X) {
				log(Level.INFO, "We don't support thread merge in X driection");	
				return false;
			}
			else {
				int myi = proc.getDefInt(GProcedure.DEF_merger_y);
				if (myi*action.getMergeNumber()>CudaConfig.getCoalescedThread()) return false;
				proc.setDef(GProcedure.DEF_merger_y, myi*action.getMergeNumber()+"");
			}
		}
		else {
			merger = new ThreadBlockMerger();			
		}
		merger.setMergeAction(action);
		merger.merge(proc);
		return true;
//		String nid = proc.getProcedure().getName().toString()+"_"+action.toString();
//		gerenateCopy(getName()+"_"+nid, proc.generateRunnableCode());
	}
	
	void reloadMemoryExpression(GProcedure proc, List<BlockDiff> blockDiffs) {
		Procedure procedure = proc.getProcedure();
		DepthFirstIterator dfi = new DepthFirstIterator(procedure);
		List<MemoryExpression> list = dfi.getList(MemoryExpression.class);
		for (BlockDiff bd: blockDiffs) {
			MemoryExpression me = bd.getMemoryExpression();
			MemoryExpression mer = bd.getMemoryExpressionR();
			for (MemoryExpression memoryExpression: list) {
				if (memoryExpression.getId()==me.getId()) {
					bd.setMemoryExpression(memoryExpression);
				}
				if (mer!=null&&memoryExpression.getId()==me.getId()) {
					bd.setMemoryExpression(memoryExpression);				
				}
			}
			
		}
		
	}

}
