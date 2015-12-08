package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;
import ece.ncsu.edu.gpucompiler.cuda.index.ConstIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.util.ExpressionEval;

/**
 * find the read after write and do thread (block) merge. 
 * For reduction
 * @author jack
 *
 */
public class RAWPass extends Pass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}
	
	class RAWMap {
		int offset;
		int writeId;
		
		@Override
		public boolean equals(Object obj) {
			RAWMap map = (RAWMap)obj;
			return (map.offset==this.offset) && (map.writeId == this.writeId);
		}

		@Override
		public String toString() {
			return offset+","+writeId+",";
		}
		
		public int hashCode(){ 
			int result; 
			result = writeId; 
			result = 29*result + offset; 
			return result; 
		} 
		
	}
	
	class Step {
		int stride = -1;
		long it;
		long activeThreads = -1;
//		List<Address> readExpression = new ArrayList();
		
		Hashtable<Integer, RAWMap> rAWMap = new Hashtable();
		Hashtable<MemoryExpression, Address> readMemoryExpressions = new Hashtable();
		
//		List<Address> writeExpression = new ArrayList();
		Hashtable<MemoryExpression, Address> writeMemoryExpressions = new Hashtable();
		
	}
	
	long power(long base, long exp, long inc) {
		long v = base;
		for (int i=0; i<exp; i++) v*= inc;
		return v;
	}
	
	void handleLoop(Statement prefix, GLoop loop) throws UnsupportedCodeException {
		/*
		 * 1. unroll for loop to steps
		 * 	step: action threads, depend thread, idx->idx, idx+stride
		 * 2. thread merge
		 * 	
		 * 3. thread block merge
		 * 
		 * 4. go back step 2
		 */		
		
		// only support *


		GProcedure proc = loop.getGProcedure();
		long start = loop.getStart();
		long inc = loop.getIncrement().getValue();
		long end = 0;
		Expression activeExpression = null;
		if (loop.getEnd() instanceof Identifier) {
			Long v = proc.getPragmasValue().get(loop.getEnd().toString());
			if (v==null) return;
			end = v.longValue();
		}
		else
		if (loop.getEnd() instanceof IntegerLiteral) {
			end = loop.getEndValue();
		}
		
		if (loop.getBody().getChildren().size()==0) return;
		Statement body = loop.getBody();
		Traversable first = loop.getBody().getChildren().get(0);
		if (first instanceof IfStatement) {
			IfStatement ifs = (IfStatement)first;
			if (ifs.getControlExpression() instanceof BinaryExpression) {
				BinaryExpression ae = (BinaryExpression)ifs.getControlExpression();
				if (ae.getLHS().toString().equals(ThreadIndex.IDX)&&
						ae.getOperator().equals(AssignmentOperator.COMPARE_LT)) {
					activeExpression = ae.getRHS();
					body = ifs.getThenStatement();
					
				}
			}
		}
		
		System.out.println("start:"+start+";inc="+inc+";end="+end);

		BreadthFirstIterator bfi = new BreadthFirstIterator(loop);

		List<MemoryExpression> memoryExpressions = bfi.getList(MemoryExpression.class);			

		long xsize = proc.getGlobalDimX();
		boolean preRead = true;
		String gMemory = null;
		for (MemoryExpression memoryExpression: memoryExpressions) {
			MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
			if (maa==null) continue;
			if (gMemory!=null&&!maa.getArrayName().toString().equals(gMemory)) {
				throw new UnsupportedCodeException("We don't support multi memory array:" + gMemory+ " & " +maa.getArrayName().toString());
			}
			gMemory = maa.getArrayName().toString();
			boolean isRead = maa==memoryExpression.getrMemoryArrayAccess();
			if (!preRead&&isRead) {
				throw new UnsupportedCodeException("We don't support read after write in loop");				
			}
		}		
		
		
		Hashtable<String, Long> values = new Hashtable();
		values.putAll(proc.getPragmasValue());
		
		
		////////////////////////////////////////////////////////////////////////////////////////
		// 1. unroll
		List<Step> steps = new ArrayList();
		Step previousStep = null;
		if (prefix!=null) {
			Step step = new Step();
			step.activeThreads = xsize;

			BreadthFirstIterator bfi_prefix = new BreadthFirstIterator(prefix);
			List<MemoryExpression> memoryExpressions_prefix = bfi_prefix.getList(MemoryExpression.class);				

			for (MemoryExpression memoryExpression: memoryExpressions_prefix) {
				MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
				if (maa==null) continue;
				Expression x=null;	// only 1D 
				x = ExpressionEval.eval(maa.getIndex(0), values);
				boolean isRead = maa==memoryExpression.getrMemoryArrayAccess();
				Address ax = Address.parseAddress(x, memoryExpression.getLoop());
				ax.compact();
//				System.out.println("put : "+ isRead + " " + memoryExpression + "," + ax.toExpression());
				if (isRead) {
//					step.readExpression.add(ax);
					step.readMemoryExpressions.put(memoryExpression, ax);
				}
				else {
//					step.writeExpression.add(ax);					
					step.writeMemoryExpressions.put(memoryExpression, ax);
				}
//				System.out.println(ax.toExpression());			
			} 
			if (previousStep!=null) {
				int stride = detect(previousStep, step);
				step.stride = stride;
			}
			steps.add(step);
			previousStep = step;
//			System.out.println("step : -1, stride="+step.stride + " active=" + step.activeThreads);
			
		}
		
		
		for (long i=start; i<end; i*=inc) {
			Step step = new Step();
			step.it = i;
			values.put(loop.getIterator().getName(), i);
			Expression activeValue = null;
			step.activeThreads = xsize;
			if (activeExpression!=null) {
				activeValue = ExpressionEval.eval(activeExpression, values);
				if (activeValue instanceof IntegerLiteral) {
					step.activeThreads =  ((IntegerLiteral)activeValue).getValue();
				}
//				System.out.println("activeValue:"+activeValue);					
			}
			for (MemoryExpression memoryExpression: memoryExpressions) {
				MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
				if (maa==null) continue;
				Expression x=null;	// only 1D 
				x = ExpressionEval.eval(maa.getIndex(0), values);
				boolean isRead = maa==memoryExpression.getrMemoryArrayAccess();
				Address ax = Address.parseAddress(x, loop);
				ax.compact();
//				System.out.println("put : "+ isRead + " " + memoryExpression + "," + ax.toExpression());
				if (isRead) {
//					step.readExpression.add(ax);
					step.readMemoryExpressions.put(memoryExpression, ax);
				}
				else {
//					step.writeExpression.add(ax);					
					step.writeMemoryExpressions.put(memoryExpression, ax);
				}
//				System.out.println(ax.toExpression());			
			} 
			if (previousStep!=null) {
				int stride = detect(previousStep, step);
				step.stride = stride;
			}
			steps.add(step);
			previousStep = step;
//			System.out.println("step : i="+i+", stride="+step.stride + " active=" + step.activeThreads);
		}
		
		
		
		
		/////////////////////////////////////////////////////////////////////////////////
		values.clear();
		values.putAll(proc.getPragmasValue());
		Step step = steps.get(0);
		Statement preBody = prefix!=null?prefix:(Statement)body.clone();
		CompoundStatement cs = new CompoundStatement();
		/*
		 * we detect the stride between steps and then do 
		 *  1. thread merge 
		 *  2. thread block merge
		 * for reduction, because stride=activeThread/2, so we only need put idx and idx+stride 
		 * into one thread, when we do thread merge. Otherwise, we need to remap the idx to
		 * bidx->bidx, bidx+stride, bidx+stride*2 ..., and coalesced number as a initial block.
		 * when we do thread block merge, we need to remap idx
		 *  
		 */
		int threadMerge = 1;
//		int reg = proc.getRegCount();
//		Statement statement = loop;
		
		int i = 0;
		List<String> sources = new ArrayList();
		

		
		while (i<steps.size()-1) {
			long j = power(start, prefix!=null?i-1:i, inc);
//			System.out.println("i"+i+",j"+j);
			Step nextstep = steps.get(i+1);
			values.put(loop.getIterator().getName(), j);
			if (nextstep.stride==-1) {
				// create new kernel
				throw new UnsupportedCodeException("failed to find data sharing");
			}
			Hashtable<RAWMap, String> addrs = new Hashtable();


			if (threadMerge<4&&proc.getRegCount()*512<CudaConfig.getDefault().getRegisterInMP()&nextstep.activeThreads>=CudaConfig.getDefault().getThreadInBlock()) {
				//&&stride>CudaConfig.getCoalescedThread()
				//&&activeThread<=512
				{
					Statement nbody = (Statement)preBody.clone();
					replaceWrite(cs, nbody, values, proc,  addrs, 0);
				}
				{
					Statement nbody = (Statement)preBody.clone();
					replaceWrite(cs, nbody, values, proc,  addrs, nextstep.stride);
				}
				
				{
					values.put(loop.getIterator().getName(), j*inc);
					Statement nbody = (Statement)body.clone();
					replaceRead(cs, nbody, values, proc,  addrs, null, null, nextstep);
				}
				Step newstep = new Step();
				newstep.activeThreads = nextstep.activeThreads;
				newstep.writeMemoryExpressions.putAll(steps.get(i+1).writeMemoryExpressions);
	
				if (nextstep.stride<nextstep.activeThreads) {
					// never happen for reduction
					Statement nbody = (Statement)body.clone();
					BinaryExpression nidx = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.ADD, new IntegerLiteral(nextstep.stride));
					replaceRead(cs, nbody, values, proc,  addrs, new Identifier(ThreadIndex.IDX), nidx, nextstep);
				}
				proc.setGlobalDimX(proc.getGlobalDimX()/2);
				preBody = cs;
				loop.swapWith(cs);			
//				new NVCCPass().dopass(proc);
				cs.swapWith(loop);
//				proc.compile();
//				System.out.println(proc.getRegCount());
				cs = new CompoundStatement();
				step = newstep;
				// current we only support stride = activeThreads/2
				// we will support more stride mode in the future
				threadMerge *= 2;
				i++;
				System.out.println("thread merge "+(i)+" to "+(i+1));
				
				
			}
			else {
				
				int prestride = 0;
				int threadBlockSize = CudaConfig.getCoalescedThread();
				if (nextstep.activeThreads<=CudaConfig.getDefault().getThreadInBlock()) {
					threadBlockSize = 1;
				}
				int baseThread = threadBlockSize;
				
				int endi = 0;
				for (int k=i+1; k<steps.size(); k+=1) {
					Step nnstep = steps.get(k);
					if (nnstep.stride==-1) {
						// new kernel
						break;
					}
					else
					if (prestride==0 || (prestride%nnstep.stride==0)) {
						threadBlockSize *= 2;
						endi = k;
						prestride = nnstep.stride;
					}
					else {
						// new kernel
						break;
					}
					
					if (threadBlockSize>CudaConfig.getDefault().getThreadInBlock()/2) {
						break;
					}
				}
				
				System.out.println("thread block merge "+(i+1)+" to "+endi+" based on "+baseThread);
				
				// remap
				Identifier nidx = null;
				AssignmentExpression nidx_replace =  null;
				if (prestride!=1||baseThread!=1) {
				// nidx = (tidx/baseThread*stride)+(idx&(baseThread-1))+(idx/blockIdx.x)*baseThread
					BinaryExpression ex = new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.DIVIDE, new IntegerLiteral(baseThread));
					ex = new BinaryExpression(ex, BinaryOperator.MULTIPLY, new IntegerLiteral(prestride));
					BinaryExpression ex1 = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.BITWISE_AND, new IntegerLiteral(baseThread-1));
					ex = new BinaryExpression(ex, BinaryOperator.ADD, ex1);
					ex1 = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.DIVIDE, new IntegerLiteral(threadBlockSize));
					ex1 = new BinaryExpression(ex1, BinaryOperator.MULTIPLY, new IntegerLiteral(baseThread));
					ex = new BinaryExpression(ex, BinaryOperator.ADD, ex1);
					nidx =  new Identifier("nidx");
					nidx_replace =  new AssignmentExpression(nidx, AssignmentOperator.NORMAL, ex);
				}
				// 
				{
					// thread block merge
					// bidx->bidx, bidx+stride, bidx+stride*2 ...
					// (idx/stride)*stride*size+(idx%stride)
					
					// merge 8

					// idx = (bidx)*coalescedNumber + (tidx%coalescedNumber) + (tidx/coalescedNumber)*stride;
					
					// idx_ = coalescedNumber*idx/stride + tidx%coalescedNumber + (bidx)*coalescedNumber
					
					// loop1: 
					
					// if (idx<4*stride)
					

					Step firstStep = steps.get(i+1);
					
					// 1. define shared memory with size, so far we only have one memory write statement
					int sharedSize = threadBlockSize*firstStep.writeMemoryExpressions.size();
					Identifier shared = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_SHARED, proc.getProcedure());
					DeclarationStatement defineShareMemoryStmt = StatementUtil.loadDefineShareMemoryStatement(shared, sharedSize, 0, Specifier.FLOAT);
					cs.addStatement(defineShareMemoryStmt);
					Tools.addSymbols(proc.getProcedure(), defineShareMemoryStmt.getDeclaration());
					
					
					replaceWriteShared(preBody, values, proc, shared, step, threadBlockSize);
					cs.addStatement(preBody);
					cs.addStatement(StatementUtil.createSynchthreadsStatement());
					// 2. update memory write to shared memory write
					for (int k=i+1; k<endi; k++) {
						Statement nbody = (Statement)body.clone();
						replaceReadShared(baseThread, nbody, values, proc, shared, steps.get(k), threadBlockSize, prestride);
						replaceWriteShared(nbody, values, proc, shared, steps.get(k), threadBlockSize);
						
						if (steps.get(k).activeThreads!=-1) {
							BinaryExpression condition = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.COMPARE_LT, new IntegerLiteral(steps.get(k).activeThreads));
							IfStatement ifs = new IfStatement(condition, nbody);
							cs.addStatement(ifs);
						}					
						else {
							cs.addStatement(nbody);
						}
						cs.addStatement(StatementUtil.createSynchthreadsStatement());
					}
					{
						
						Statement nbody = (Statement)body.clone();
						replaceReadShared(baseThread, nbody, values, proc, shared, steps.get(endi), threadBlockSize, prestride);
						if (steps.get(endi).activeThreads!=-1) {
							BinaryExpression condition = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.COMPARE_LT, new IntegerLiteral(steps.get(endi).activeThreads));
							IfStatement ifs = new IfStatement(condition, nbody);
							cs.addStatement(ifs);
						}					
						else {
							cs.addStatement(nbody);
						}
					}
					
					i = endi+1;
					proc.setBlockDimX(threadBlockSize);
					if (nidx!=null) {
						CetusUtil.replaceChild(cs, new Identifier(ThreadIndex.IDX), nidx);
						StatementUtil.addToFirst(cs, new ExpressionStatement(nidx_replace));
						StatementUtil.addToFirst(cs, StatementUtil.loadIntDeclaration(nidx));
					}
//					System.out.println(cs);
					loop.swapWith(cs);
					String source = proc.generateRunnableCode();
					sources.add(source);
					cs.swapWith(loop);
					
//					System.out.println("--------------------kernel--------------------");
//					System.out.println(source);
					// 3. update memory read to shared memory read
					
					if (endi+1<steps.size())
						proc.setGlobalDimX((int)steps.get(endi+1).activeThreads);
					
					// we need on new kernel
					cs = new CompoundStatement();
					threadMerge = 1;
					preBody = (Statement)body.clone();
					
				}
				
			}

		}
		
		String funcName = proc.getProcedure().getName().toString();
		for (String s: sources) {
			proc.refresh(s);
			proc.getProcedure().setName(new Identifier(funcName+"_"+sources.indexOf(s)));
			proc.gerenateOutput(funcName+"_"+sources.indexOf(s)+"_output");
		}
//		proc.refresh();
//		System.out.println(proc.getProcedure());
		
	}

	
	void replaceReadShared(int baseThread, Statement nbody, Hashtable<String, Long> values, GProcedure proc, Identifier shared, Step step, int tbsize, int stride) throws UnsupportedCodeException {

		for (String key : values.keySet()) {
			CetusUtil.replaceChild(nbody, new Identifier(key), new IntegerLiteral(values.get(key)));					
		}

		List<MemoryExpression> mes = new BreadthFirstIterator(nbody).getList(MemoryExpression.class);			

		for (MemoryExpression me: mes) {
			if (me.getGlobalMemoryArrayAccess()!=null&&me.getGlobalMemoryArrayAccess()==me.getrMemoryArrayAccess()) {
				RAWMap map = step.rAWMap.get(me.getId());
				if (map!=null) {
					int offset = 0;
					for (MemoryExpression m: step.writeMemoryExpressions.keySet()) {
						if (map.writeId==m.getId()) {
							break;
						}
						offset += tbsize;
//						System.out.println("offset="+offset+":"+m+"");
					}
					Identifier id = new Identifier(ThreadIndex.TIDX);
					if (map.offset<CudaConfig.getCoalescedThread()) {
//						id = new Identifier("ctidx");
						offset += map.offset;
					}
					else {
//						id = new Identifier("ctidx");
						offset += map.offset/(stride/baseThread);
						
						
					}
//					System.out.println("offset="+offset+":"+me);
					ArrayAccess tmp = new ArrayAccess((Identifier)shared.clone(), new BinaryExpression(id, BinaryOperator.ADD, new IntegerLiteral(offset)));
					AssignmentExpression ae =  new AssignmentExpression(me.getLHS(), me.getOperator(), tmp);
					me.swapWith(ae);
				}
			}
		}
		
//		if (nbody.getParent()!=cs)
//			if (step.activeThreads!=-1) {
//				BinaryExpression condition = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.COMPARE_LT, new IntegerLiteral(step.activeThreads));
//				IfStatement ifs = new IfStatement(condition, nbody);
//				cs.addStatement(ifs);
//			}
//			else {
//				cs.addStatement(nbody);
//			}		
	}
	
	void replaceWriteShared(Statement nbody, Hashtable<String, Long> values, GProcedure proc, Identifier shared, Step step, int tbsize) throws UnsupportedCodeException {

		for (String key : values.keySet()) {
			CetusUtil.replaceChild(nbody, new Identifier(key), new IntegerLiteral(values.get(key)));					
		}

		List<MemoryExpression> mes = new BreadthFirstIterator(nbody).getList(MemoryExpression.class);			

		
		int i = 0;
		for (MemoryExpression me: mes) {
			if (me.getGlobalMemoryArrayAccess()!=null&&me.getGlobalMemoryArrayAccess()==me.getlMemoryArrayAccess()) {
				ArrayAccess tmp = new ArrayAccess((Identifier)shared.clone(), new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.ADD, new IntegerLiteral(i)));
				AssignmentExpression ae =  new AssignmentExpression(tmp, me.getOperator(), me.getRHS());
				me.swapWith(ae);
				i += tbsize;
			}
		}
	}

	void replaceWrite(CompoundStatement cs, Statement nbody, Hashtable<String, Long> values, GProcedure proc, Hashtable<RAWMap, String> addrs, int stride) throws UnsupportedCodeException {
		if (stride!=0) {
			BinaryExpression nidx = new BinaryExpression(new Identifier(ThreadIndex.IDX), BinaryOperator.ADD, new IntegerLiteral(stride));
			CetusUtil.replaceChild(nbody, new Identifier(ThreadIndex.IDX), nidx);				
		}

		for (String key : values.keySet()) {
			CetusUtil.replaceChild(nbody, new Identifier(key), new IntegerLiteral(values.get(key)));					
		}

		List<MemoryExpression> mes = new BreadthFirstIterator(nbody).getList(MemoryExpression.class);			

		for (MemoryExpression me: mes) {
			if (me.getGlobalMemoryArrayAccess()!=null&&me.getGlobalMemoryArrayAccess()==me.getlMemoryArrayAccess()) {
				MemoryArrayAccess maa = me.getGlobalMemoryArrayAccess();
				maa.setIndex(0, ExpressionEval.eval(maa.getIndex(0), new Hashtable()));
				maa.load();
				Address x = maa.getX();
				x.compact();
				Identifier tmp = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_TMP, proc.getProcedure());
				AssignmentExpression ae =  new AssignmentExpression((Identifier)tmp.clone(), me.getOperator(), me.getRHS());
				me.swapWith(ae);

				DeclarationStatement ds = StatementUtil.loadInitStatment(tmp, Specifier.FLOAT);
				Tools.addSymbols(proc.getProcedure(), ds.getDeclaration());
				cs.addStatement(ds);
				RAWMap map = new RAWMap();
				map.offset = stride;
				map.writeId = me.getId();
//				System.out.println("put " + me.getId() + "" + ae + "," + map.offset + "," + map.writeId);
				addrs.put(map, tmp.toString());
			}
		}
		cs.addStatement(nbody);
	}

	
	
	void replaceRead(CompoundStatement cs, Statement nbody, Hashtable<String, Long> values, GProcedure proc, Hashtable<RAWMap, String> addrs, Expression oid, Expression nid, Step nextStep) throws UnsupportedCodeException {		
		if (oid!=null&&nid!=null)
			CetusUtil.replaceChild(nbody, oid, nid);				
		for (String key : values.keySet()) {
			CetusUtil.replaceChild(nbody, new Identifier(key), new IntegerLiteral(values.get(key)));					
		}
		
		List<MemoryExpression> mes = new BreadthFirstIterator(nbody).getList(MemoryExpression.class);			

		for (MemoryExpression me: mes) {
//			System.out.println(me+" replaceRead id "+ me.getId());
			if (me.getGlobalMemoryArrayAccess()!=null&&me.getGlobalMemoryArrayAccess()==me.getrMemoryArrayAccess()) {
				RAWMap raw = nextStep.rAWMap.get(me.getId());
				if (raw!=null) {
//					System.out.println(addrs.toString());
					String tmp = addrs.get(raw);
//					System.out.println(me+" replaceRead id "+ raw.offset + "," + raw.writeId + ","+ tmp);
					if (tmp!=null) {
						AssignmentExpression ae =  new AssignmentExpression(me.getLHS(), me.getOperator(), new Identifier(tmp));
						me.swapWith(ae);
					}
					else {
					}
				}
//				System.out.println("out:"+x.toExpression());
			}
		}				
		cs.addStatement(nbody);		
	}
	
	/**
	 * find the stride to merge threads
	 * the memory write in step should be satisfy the memory read in nextStep
	 * @param step
	 * @param nextStep
	 * @return
	 */
	int detect(Step step, Step nextStep) {
		int stride = 0;
		
		Hashtable<MemoryExpression, Address> writes = step.writeMemoryExpressions;
		Hashtable<MemoryExpression, Address> reads = nextStep.readMemoryExpressions;
		
		for (MemoryExpression mread: reads.keySet()) {
			boolean isfind = false;
			Address read = reads.get(mread);
//			System.out.println("detect: " + read.toExpression());
			for (MemoryExpression mwrite: writes.keySet()) {
				Address write = writes.get(mwrite);
				Address addr = (Address)read.clone();
				addr = addr.subtract(write);
				isfind = addr.isZero();
				if (isfind) {
					RAWMap map = new RAWMap();
					map.offset = 0;
					map.writeId = mwrite.getId();
					nextStep.rAWMap.put(mread.getId(), map);
//					System.out.println("match: " + mread+" : "+mwrite);
					break;
				}
			}
			
			if (!isfind) {
				if (read.isContain(ThreadIndex.getThreadIndex(ThreadIndex.IDX)).size()!=1) return -1;
				ThreadIndex idxR = (ThreadIndex)read.isContain(ThreadIndex.getThreadIndex(ThreadIndex.IDX)).get(0);
//				System.out.println("detect: stride "+writes.keySet().size());
				for (MemoryExpression mwrite: writes.keySet()) {
					Address write = writes.get(mwrite);
//					System.out.println("detect: "+write.toExpression());
					if (write.isContain(ThreadIndex.getThreadIndex(ThreadIndex.IDX)).size()!=1) continue;
					ThreadIndex idxW = (ThreadIndex)write.isContain(ThreadIndex.getThreadIndex(ThreadIndex.IDX)).get(0);
					if (idxR.getCoefficient()!=idxW.getCoefficient()) continue;
					Address addr = (Address)read.clone();
					addr = addr.subtract(write);
					if (addr.getIndexs().size()==1) {
						Index in = addr.getIndexs().get(0);
						if (in instanceof ConstIndex) {
							int stride0 = in.getCoefficient()/idxR.getCoefficient();
//							System.out.println("match: " + stride0+" : "+mwrite);
							RAWMap map = new RAWMap();
							map.offset = stride0;
							map.writeId = mwrite.getId();
							nextStep.rAWMap.put(mread.getId(), map);
							if (stride!=0&&stride0!=stride) {
								return -1;
							}
							stride = stride0;
						}
					}
					
				}
				if (stride==0) return -1;
			}
		}
		
		return stride;
	}
	
	@Override
	public void dopass(GProcedure proc) throws UnsupportedCodeException {
		/**
		 * 1. first find the top level loop
		 * 2. find the active thread
		 * 3. find memory statement
		 */
		BreadthFirstIterator bfi = new BreadthFirstIterator(proc.getProcedure());

		List<GLoop> gloops = bfi.getList(GLoop.class);	
		for (GLoop loop: gloops) {
			if (loop.isSimpleLoop()) continue;
			if (loop.getParent()!=proc.getProcedure().getBody()) continue;
			if (!loop.getIncType().equals(BinaryOperator.MULTIPLY)) continue;
			// if not synchronization
			int i = loop.getParent().getChildren().indexOf(loop);
			i = i-1;
			CompoundStatement prefix = new CompoundStatement();
			for (int j=i; j>=0; j--) {
				if (loop.getParent().getChildren().get(j) instanceof Statement) {
					Statement st = (Statement)loop.getParent().getChildren().get(j);
					if (st instanceof DeclarationStatement) break;
					st.detach();
//					System.out.println("prefix:"+j+"-"+st);
					StatementUtil.addToFirst(prefix, st);
				}
			}
//			System.out.println(proc.getProcedure());
//			StatementUtil.addSibling(loop, prefix, true);
//			System.out.println(proc.getProcedure());
			handleLoop(prefix, loop);
		}
		
		

	}

	
}
