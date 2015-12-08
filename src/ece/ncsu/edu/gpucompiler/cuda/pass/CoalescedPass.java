package ece.ncsu.edu.gpucompiler.cuda.pass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DeclarationStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Tools;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CetusUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.CudaConfig;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GProcedure;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArray;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryArrayAccess;
import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;
import ece.ncsu.edu.gpucompiler.cuda.cetus.StatementUtil;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;
import ece.ncsu.edu.gpucompiler.cuda.cetus.VariableTools;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;
import ece.ncsu.edu.gpucompiler.cuda.index.ConstIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.Index;
import ece.ncsu.edu.gpucompiler.cuda.index.LoopIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.ThreadIndex;
import ece.ncsu.edu.gpucompiler.cuda.index.UnresolvedIndex;

public class CoalescedPass extends Pass {

	@Override
	public String getName() {
		return this.getClass().getSimpleName();
	}	
	
	

	
	protected static boolean isCoalesced(Address addressx, Address addressy) {
		if (addressy!=null)
		for (Index index: addressy.getIndexs()) {
			if (index instanceof ThreadIndex) {
				// y coordinator cannot include idx
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.IDX)) return false;
			}
		}
		boolean iscolaescd = false;
//		System.out.println("find:"+addressx);
		for (Index index: addressx.getIndexs()) {
			if (index instanceof ConstIndex) {
				ConstIndex ci = (ConstIndex)index;
				// constant value only can be 16x
				if (ci.getCoefficient()%CudaConfig.getCoalescedThread()!=0) return false;
			}
			else 
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.IDX)&&ti.getCoefficient()==1) {
					iscolaescd = true;				
				}
			}
			else
			if (index instanceof LoopIndex) {
				// TODO: it's possible coalesced
//				LoopIndex li = (LoopIndex)index;
//				System.out.println("find:"+index);
				return false;
			}
		}
		return iscolaescd;
	}	
	
	@SuppressWarnings("unchecked")
	public void dopass(GProcedure proc) throws UnsupportedCodeException {
		int coalescedNumber = CudaConfig.getCoalescedThread();
		// first
		proc.setBlockDimX(coalescedNumber);
		
		boolean haschange = true;
		while (haschange) {
			haschange = false;
			DepthFirstIterator dfi = new DepthFirstIterator(proc.getProcedure());
			List<MemoryExpression> memoryExpressions = (List<MemoryExpression>)dfi.getList(MemoryExpression.class);
			for (MemoryExpression memoryExpression: memoryExpressions) {
//				System.out.println(memoryExpression.toInfoString());

				if (memoryExpression.getSharedMemoryArrayAccess()!=null) {
					continue;
				}
				if (memoryExpression.getGlobalMemoryArrayAccess()==memoryExpression.getlMemoryArrayAccess()) {
					log(Level.INFO, memoryExpression+": is global memory write, ignore it");
					continue;					
				}
				
				if (memoryExpression.getGlobalMemoryArrayAccess()!=null) {
	//				System.out.println(memoryExpression.getGlobalMemoryArrayAccess());
					MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
					boolean coalseced = false;
					if (maa.getNumIndices()==2) { 
						coalseced = isCoalesced(maa.getIndexAddress(1), maa.getIndexAddress(0));
					}
					else {
						coalseced = isCoalesced(maa.getIndexAddress(0), null);						
					}
					if (coalseced) {
						log(Level.INFO, memoryExpression+":already coalesced");
						continue;
					}
					else {
						
						String id = memoryExpression.toString();
						Snapshot ss = new Snapshot();
						ss.setId(id);
						ss.setId(getName());
						dopass(memoryExpression);
						ss.setProcedure(proc.generateRunnableCode());
						// we can roll back and disbale these memory coalesced
						// for example, in the thread merge stage, we set 
						// Snapshot.setProperty("disable", "true");
						// then at this stage, we check it
						proc.getHistory().add(ss);
						haschange = true;
						String nid = id;
						proc.gerenateOutput(proc.getProcedure().getName().toString()+"_"+getName()+"_"+nid);
						break;
					}
				}
			}
			if (haschange)
				proc.refresh();
		}

	}
	
	public void dopass(MemoryExpression memoryExpression) {
		
		
		
		log(Level.INFO, memoryExpression+":not coalesced");
		try {
			doConvert(memoryExpression);
		} catch (UnsupportedCodeException e) {
			log(Level.INFO, memoryExpression+": converting failed: "+e.getMessage());
			e.printStackTrace();
		}
		
	}
	
	class Offset {
		int yoffset;
		// follow are x offset
		int baseOffset;
		int ioffset;
		int idxoffset;
		int sharedMemorySize;
		int offset_min;
	}
	
	Offset computeOffset(MemoryExpression memoryExpression) {
		int coalescedNumber = CudaConfig.getCoalescedThread();
		
		Address x, y;
		MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
		x = maa.getX();
		y = maa.getY();
		
		
		int yoffset = 0;
		//$$$2. compute the offset for Y
		if (y!=null) {
			ArrayList<Index> indexs = y.isContain(new ThreadIndex(ThreadIndex.IDX));
			for (Index in: indexs) {
				yoffset += in.getCoefficient();
			}
			log(Level.INFO, memoryExpression+": offset of Y is "+ yoffset);
		}
		
		
		//$$$3. compute the offset for X
		int baseOffset = 0, ioffset = 0, idxoffset = 0;
		//$$$3.1 baseOffset: compute the offset for the first thread and first iterator
		//$$$3.2 ioffset: compute the offset for iterator
		//$$$3.3 idxoffset: compute the offset for idx
		// if baseOffset = 0, ioffset = 0, idxoffset = 1 then coalesced
		for (Index ind : x.getIndexs()) {
			if (ind instanceof ConstIndex) {
				ConstIndex ci = (ConstIndex) ind;
				baseOffset += ci.getCoefficient()%coalescedNumber;
			} 
			else if (ind instanceof ThreadIndex) {
				ThreadIndex in = (ThreadIndex)ind;
				if (in.getId().equals(ThreadIndex.IDX)) {
					idxoffset += in.getCoefficient();
				}
			} 
			else if (ind instanceof LoopIndex) {
				LoopIndex li = (LoopIndex) ind;
				long start = li.getLoop().getStart();
				baseOffset += li.getCoefficient()*start;
				ioffset += li.getCoefficient()*li.getLoop().getIncrement().getValue();
			}
			else if (ind instanceof UnresolvedIndex) {
				UnresolvedIndex in = (UnresolvedIndex) ind;
				log(Level.INFO, memoryExpression+": cannot recognize ["+in.getId()+"] and assume it multiple of "+coalescedNumber);
			}
		}	
		
		int end = memoryExpression.getLoop().getEndValue();
		if (end==Integer.MAX_VALUE) {
			end = coalescedNumber;
		}
		int ioffset_max = ioffset*(end-1);
		int idxoffset_max = idxoffset*(coalescedNumber-1);
		int offset_0 = ioffset_max+idxoffset_max;
		int offset_1 = 0;
//		if (ioffset_max*idxoffset_max<0) {
//			offset_0 = ioffset_max-idxoffset_max;
//			offset_1 = -offset_0;
//		}
		int offset_2 = ioffset_max;
		int offset_3 = idxoffset_max;
		int offset_max = Math.max(Math.max(offset_0, offset_1), Math.max(offset_2, offset_3));
		int offset_min = Math.min(Math.min(offset_0, offset_1), Math.min(offset_2, offset_3));
		offset_min += baseOffset;
		offset_max += baseOffset;
		offset_min = CetusUtil.getSmallMultiple(offset_min, coalescedNumber);
//		offset_max = (offset_max%coalescedNumber==0)? offset_max-coalescedNumber : CetusUtil.getSmallMultiple(offset_max, coalescedNumber);
		offset_max = CetusUtil.getSmallMultiple(offset_max, coalescedNumber);
		
//		int offset = offset_max-offset_min;
//		
//		int offset_segment = offset;
		
		System.out.println("offset_min"+offset_min+"offset_max"+offset_max);
		Offset offset = new Offset();
		offset.yoffset = yoffset;
		offset.idxoffset = idxoffset;
		offset.baseOffset = baseOffset;
		offset.ioffset = ioffset;
		offset.sharedMemorySize = offset_max-offset_min+coalescedNumber;
		if (offset.sharedMemorySize==0) offset.sharedMemorySize = coalescedNumber;
		offset.offset_min = offset_min;
		return offset;
	}
	
	

	
	// define the shared memory
	Identifier defineSharedMemory(MemoryExpression memoryExpression, int xsize, int ysize) {
		GProcedure proc = memoryExpression.getLoop().getGProcedure();
		MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
		
		Identifier shared = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_SHARED, proc.getProcedure());
		DeclarationStatement defineShareMemoryStmt = StatementUtil.loadDefineShareMemoryStatement(shared, xsize, ysize, maa.getMemoryArray().getType());
		MemoryArray ma = new MemoryArray(shared.toString(), ysize==0?1:2, maa.getMemoryArray().getType(), MemoryArray.MEMORY_SHARED);
		
		proc.addMemoryArray(ma);

		Statement firststmtforproc = (Statement) StatementUtil.getFirstDeclarationStatement(proc.getProcedure().getBody());
		StatementUtil.addSibling(firststmtforproc, defineShareMemoryStmt, true);
		Tools.addSymbols(proc.getProcedure(), defineShareMemoryStmt.getDeclaration());
		
		return shared;
	}
	
	
	CompoundStatement mapCoalescedStartAddress(Offset offset, Identifier shared, MemoryExpression memoryExpression, boolean is2D) throws UnsupportedCodeException {
		int coalescedNumber = CudaConfig.getCoalescedThread();
		GProcedure proc = memoryExpression.getLoop().getGProcedure();
		Identifier it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, proc.getProcedure());
		MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
		Address x = maa.getX();
		Address y = maa.getY();		

		// based address:
		// idx->idx-tidx
		// addr = addr-baseOffset
		Address globalAddrx = new Address();
		Address globalAddry = new Address();
		
		// we add baseOffset
		globalAddrx.getIndexs().add(new ConstIndex(-offset.baseOffset));
		for (Index ind : x.getIndexs()) {
			if (ind instanceof ThreadIndex) {
				ThreadIndex in = (ThreadIndex)ind.clone();
				if (in.getId().equals(ThreadIndex.IDX)) {
					// idx -> idx-tidx
					ThreadIndex idx = ThreadIndex.getThreadIndex(ThreadIndex.IDX);
					idx.setCoefficient(in.getCoefficient());
					ThreadIndex tidx = ThreadIndex.getThreadIndex(ThreadIndex.TIDX);
					tidx.setCoefficient(-in.getCoefficient());
					globalAddrx.getIndexs().add(idx);
					globalAddrx.getIndexs().add(tidx);
				}
				else
				if (in.getId().equals(ThreadIndex.IDY)) {
					in.setId(ThreadIndex.COALESCED_IDY);
					globalAddrx.getIndexs().add(in);
				}					
			} 
			else 
			if (ind instanceof LoopIndex) {
				if (offset.ioffset!=0) {
					globalAddrx.getIndexs().add((Index)ind.clone());
				}
//				LoopIndex li = new LoopIndex(it);
//				li.setCoefficient(ind.getCoefficient());
//				globalAddrx.getIndexs().add(li);
			}
			else {
				globalAddrx.getIndexs().add((Index)ind.clone());
			}
		}	
		if (offset.ioffset<0) {
			globalAddrx.getIndexs().add(new ConstIndex(offset.ioffset*coalescedNumber));
		}
		
		if (y!=null) {
			for (Index ind : y.getIndexs()) {
				if (ind instanceof ThreadIndex) {
					ThreadIndex ti = (ThreadIndex)ind.clone();
					// idx->(idx-tidx)+it
					if (ti.getId().equals(ThreadIndex.IDX)) {
						globalAddry.getIndexs().add(ti);
						ThreadIndex ti0 = new ThreadIndex(ThreadIndex.TIDX);
						ti0.setCoefficient(-ti.getCoefficient());
						globalAddry.getIndexs().add(ti0);
						LoopIndex li = new LoopIndex(it);
						li.setCoefficient(ti.getCoefficient());
						globalAddry.getIndexs().add(li);
					}
					else {
						globalAddry.getIndexs().add(ti);						
					}
				}
				else {
					globalAddry.getIndexs().add((Index)ind.clone());						
				}
			}			
		}
		
		
		// then let it be coalesced
		globalAddrx.getIndexs().add(ThreadIndex.getThreadIndex(ThreadIndex.TIDX));	
		CompoundStatement gloabl2SharedStmt =  new CompoundStatement();
		if (is2D) {
			List<Expression> indics = new ArrayList<Expression>();
			indics.add(it);
			indics.add(new Identifier(ThreadIndex.TIDX));
			ArrayAccess shareArray = null;
			shareArray = new ArrayAccess(shared, indics);		 
			
			ArrayAccess globalArray = (ArrayAccess)maa.clone();
			globalArray.setIndex(0, globalAddry.toExpression());	
			globalArray.setIndex(1, globalAddrx.toExpression());
			AssignmentExpression ae = new AssignmentExpression(shareArray,
					AssignmentOperator.NORMAL, (Expression)globalArray.clone());
			ForLoop forloop = StatementUtil.loadSimpleLoop(new ExpressionStatement(ae), it, 0, 1, coalescedNumber);
			GLoop gloop =  new GLoop(forloop, proc);
			MemoryExpression me =  new MemoryExpression(ae, gloop);
			me.swapWith(ae);
			DeclarationStatement defit = StatementUtil.loadInitStatment(it, Specifier.INT);
			Tools.addSymbols(proc.getProcedure(), defit.getDeclaration());
			gloabl2SharedStmt.addStatement(defit);
			gloabl2SharedStmt.addStatement(gloop);
		}
		else {
			for (int i=0; i<offset.sharedMemorySize/coalescedNumber; i++) {
				Address addr = (Address)globalAddrx.clone();
				ConstIndex ci = new ConstIndex(0);
				ci.setCoefficient(i*coalescedNumber);
				addr.getIndexs().add(ci);
				BinaryExpression be = new BinaryExpression(new Identifier(ThreadIndex.TIDX), BinaryOperator.ADD, new IntegerLiteral(i*coalescedNumber));
				ArrayAccess sharedAddr = new ArrayAccess(shared, be);
				ArrayAccess globalAddr = null;
				if (globalAddry.getIndexs().size()!=0) {
					globalAddr = new ArrayAccess(maa.getArrayName(), globalAddry.toExpression());
					globalAddr.addIndex(addr.toExpression());
				}
				else {
					globalAddr = new ArrayAccess(maa.getArrayName(), addr.toExpression());
				}
				AssignmentExpression ae = new AssignmentExpression(sharedAddr,
						AssignmentOperator.NORMAL, globalAddr);
				gloabl2SharedStmt.addStatement(new ExpressionStatement(ae));
				
			}
		}
		gloabl2SharedStmt.addStatement(StatementUtil.createSynchthreadsStatement());
		return gloabl2SharedStmt;
	}
	
	
	Statement mapGlobal2Shared(Offset offset, MemoryExpression memoryExpression, Identifier shared) throws UnsupportedCodeException {
		GProcedure proc = memoryExpression.getLoop().getGProcedure();
		Identifier it = VariableTools.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, proc.getProcedure());
		Address shareAddrX = new Address();
		Address shareAddrY = new Address();

		MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
		Address x = maa.getX();
		Address y = maa.getY();
		if (y!=null) {
			for (Index index: y.getIndexs()) {
				if (index instanceof ThreadIndex) {
					ThreadIndex ti = (ThreadIndex)index;
					if (ti.getId().equals(ThreadIndex.IDX)) {
						// only idx contribute to the yoffset
						ThreadIndex ti0 =  new ThreadIndex(ThreadIndex.TIDX);
						ti0.setCoefficient(ti.getCoefficient());
						shareAddrY.getIndexs().add(ti0);
					}
				}
			}
		}
		for (Index index: x.getIndexs()) {
			// loopindex contribute to the ioffset
			// idx contribute to the idxoffset
			// and we need to get the offset in the shared memory
			if (index instanceof LoopIndex) {
				if (((LoopIndex) index).getId().equals(memoryExpression.getLoop().getIterator())) {
					// ioffset!=0
					if (offset.ioffset!=0) {
						LoopIndex in = new LoopIndex((Identifier)it.clone());
						in.setCoefficient(index.getCoefficient());
						shareAddrX.getIndexs().add(in);
					}
				}
			}
			else 
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.IDY)) {
					//TODO: I don't remember the effect of this code
					ThreadIndex ti0 =  new ThreadIndex(ThreadIndex.COALESCED_IDY);
					ThreadIndex ti1 =  new ThreadIndex(ThreadIndex.IDY);
					ti1.setCoefficient(ti.getCoefficient());
					ti0.setCoefficient(-ti.getCoefficient());
					shareAddrX.getIndexs().add(ti1);
					shareAddrX.getIndexs().add(ti0);
				}
				else
				if (ti.getId().equals(ThreadIndex.IDX)) {
					ThreadIndex ti0 =  new ThreadIndex(ThreadIndex.TIDX);
					ti0.setCoefficient(ti.getCoefficient());
					shareAddrX.getIndexs().add(ti0);
				}					
			}
		}		
		shareAddrX.getIndexs().add(new ConstIndex(-offset.offset_min));	
		
		List<Expression> slist = new ArrayList<Expression>();
		if (shareAddrY.getIndexs().size()>0) {
			slist.add(shareAddrY.toExpression());
		}
		slist.add(shareAddrX.toExpression());
		ArrayAccess shareReadArray = new ArrayAccess(shared, slist);
		maa.swapWith(shareReadArray);
		System.out.println("new access:"+shareReadArray);
//		CetusUtil.replace(maa, shareReadArray);	
		Statement stmt;
		if (offset.ioffset!=0) {
			// we need to unroll
			GLoop gloop = unrollLoopForIOffset(memoryExpression, it);
			stmt = gloop;
			
		}
		else {
			stmt = memoryExpression.getStatement();
			StatementUtil.addSibling(stmt, StatementUtil.createSynchthreadsStatement(), false);		
		}		
		StatementUtil.addSibling(stmt, StatementUtil.createSynchthreadsStatement(), false);		
		return stmt;
	}
	
	// if the offset.ioffset!=0, we need to unroll the loop
	GLoop unrollLoopForIOffset(MemoryExpression memoryExpression, Identifier it) throws UnsupportedCodeException {
		int coalescedNumber = CudaConfig.getCoalescedThread();
		GProcedure proc = memoryExpression.getLoop().getGProcedure();

		long inc = memoryExpression.getLoop().getIncrement().getValue();
		memoryExpression.getLoop().setIncrement(new IntegerLiteral(inc*coalescedNumber));
//		System.out.println("memoryExpression:"+memoryExpression);
//		System.out.println("memoryExpression:"+memoryExpression.getLoop());
		
		Statement stmt = memoryExpression.getLoop().getBody();
		
		DeclarationStatement defit = StatementUtil.loadInitStatment(it, Specifier.INT);
		Statement stmt1= (Statement)stmt.clone();
		CetusUtil.replaceChild(stmt1, memoryExpression.getLoop().getIterator(), new BinaryExpression(it, BinaryOperator.ADD, memoryExpression.getLoop().getIterator()));
		int end = memoryExpression.getLoop().getEndValue();
		int step = (int)((end - memoryExpression.getLoop().getStart())/inc);		
		ForLoop forloop = StatementUtil.loadSimpleLoop(stmt1, it, 0, (int)inc, step>coalescedNumber?coalescedNumber:step);
		GLoop gloop =  new GLoop(forloop, proc);
		gloop.setLoopParent(memoryExpression.getLoop());
		CompoundStatement cs = new CompoundStatement();
		cs.addStatement(defit);
		cs.addStatement(gloop);
		stmt.swapWith(cs);		
		Tools.addSymbols(proc.getProcedure(), defit.getDeclaration());
		return gloop;
	}
	
	
	
	int computePadding(Offset offset) {
		// thread i in bank: ((coalescedNumber+padding)*i + idxoffset*i)%coalescedNumber
		// -> ((padding+ idxoffset)*i)%coalescedNumber
		// -> let (padding+idxoffset) be odd
		int padding  = ((offset.idxoffset%2)==0)?1:0;
		if (offset.idxoffset==0) {
			padding = (offset.ioffset!=0)?1:0;
		}
		return padding;
	}
	
	
	void do1DShared(Offset offset, MemoryExpression memoryExpression) throws UnsupportedCodeException {
		//$$$ yoffset==0
		//		addr0 = (baseOffset);
		//		addr0 = addr0 - (addr0%coalescedNumber);
		//		addr1 = baseOffset+(ioffset+idxoffset)*15;
		//		addr1 = addr1 - (addr0%coalescedNumber);
		//		load [][addr0:addr1+15]
		
		// 1. compute the size of shared memory and define shared memory
		Identifier shared = defineSharedMemory(memoryExpression, offset.sharedMemorySize, 0);
		

		// 2. change global memory load to share memory load
		Statement currentStmt = mapGlobal2Shared(offset, memoryExpression, shared);

		// 3. global address to shared address
		// we compute the start address of global memory
		CompoundStatement gloabl2SharedStmt = mapCoalescedStartAddress(offset, shared, memoryExpression, false);
		StatementUtil.addSibling(currentStmt, gloabl2SharedStmt, true);
	}
	
	
	void do2DShared(Offset offset, MemoryExpression memoryExpression) throws UnsupportedCodeException {
		int coalescedNumber = CudaConfig.getCoalescedThread();

		
		//$$$4. yoffset!=0, we ignore ioffset now
		//	For (i=0; i<15; i++) {
		//		addr = baseOffset+idxoffset*i;
		//		addr = addr - (addr%coalescedNumber);
		//		load [i*yoffset][addr:addr+15]
		//	}
		
		// 1. define shared is 2D
//		int padding  = computePadding(offset);
		Identifier shared = defineSharedMemory(memoryExpression, coalescedNumber+1, coalescedNumber);
		
		

		// 2. change global memory load to share memory load
		Statement currentStmt = mapGlobal2Shared(offset, memoryExpression, shared);
		
		
		// 3. load global memory to shared memory
		CompoundStatement gloabl2SharedStmt = mapCoalescedStartAddress(offset, shared, memoryExpression, true);
		StatementUtil.addSibling(currentStmt, gloabl2SharedStmt, true);
		
//		Identifier it = CetusUtil.getUnusedSymbol(CetusUtil.SYMBOL_PREFIX_ITERATOR, proc.getProcedure());
//		DeclarationStatement defit = StatementUtil.loadInitStatment(it, Specifier.INT);
//
//		Address globalAddrx = (Address)x.clone();
//        globalAddrx.getIndexs().add(new ThreadIndex(ThreadIndex.TIDX));	// coalesced load
//		for (Index index: globalAddrx.getIndexs()) {
//			if (index instanceof ThreadIndex) {
//				ThreadIndex ti = (ThreadIndex)index;
//				if (ti.getId().equals(ThreadIndex.IDY)) {
//					ti.setId(ThreadIndex.COALESCED_IDY);
//				}
//			}
//			
//		}
//		ArrayAccess globalArray = (ArrayAccess)maa.clone();
//		globalArray.setIndex(1, globalAddrx.toExpression());
//		
//
//		Address globalAddrY = (Address)y.clone();
//		for (int t=0; t<globalAddrY.getIndexs().size(); t++) {
//			Index index = globalAddrY.getIndexs().get(t);
//			if (index instanceof ThreadIndex) {
//				ThreadIndex ti = (ThreadIndex)index;
//				// idx->(idx-tidx)+it
//				if (ti.getId().equals(ThreadIndex.IDX)) {
//					ThreadIndex ti0 = new ThreadIndex(ThreadIndex.TIDX);
//					ti0.setCoefficient(-ti.getCoefficient());
//					globalAddrY.getIndexs().add(ti0);
//					LoopIndex li = new LoopIndex(it);
//					li.setCoefficient(ti.getCoefficient());
//					globalAddrY.getIndexs().add(li);
//				}
//			}
//			
//		}
//		globalArray.setIndex(0, globalAddrY.toExpression());	
//
//		// shared memory access
//		List<Expression> indics = new ArrayList<Expression>();
//		indics.add(it);
//		indics.add(new Identifier(ThreadIndex.TIDX));
//		ArrayAccess shareArray = null;
//		shareArray = new ArrayAccess(shared, indics);		 
//		
//		AssignmentExpression ae = new AssignmentExpression(shareArray,
//				AssignmentOperator.NORMAL, (Expression)globalArray.clone());
//		ForLoop forloop = StatementUtil.loadSimpleLoop(new ExpressionStatement(ae), it, 0, 1, coalescedNumber);
//		GLoop gloop =  new GLoop(forloop, proc);
//		MemoryExpression me =  new MemoryExpression(ae, gloop);
//		me.swapWith(ae);
//		
//		CetusUtil.addSibling(currentStmt, defit, true);
//		CetusUtil.addSibling(defit, gloop, false);
//		CetusUtil.addSibling(gloop, CetusUtil.createSynchthreadsStatement(), false);
//		
//		
//		
//		
//		System.out.println(proc.getProcedure());
//		
	}
	
	void doConvert(MemoryExpression memoryExpression) throws UnsupportedCodeException {

//		GProcedure proc = memoryExpression.getLoop().getGProcedure();
		// $$$1. 1D, 2D, or 3D
		MemoryArrayAccess maa = memoryExpression.getGlobalMemoryArrayAccess();
		if (maa.getNumIndices()==3) { // 3D
			throw new UnsupportedCodeException("we don't support 3D array now");
		}

		
		Offset offset = computeOffset(memoryExpression);
		
		log(Level.INFO, memoryExpression+": offset of X is baseOffset("+ offset.baseOffset+")ioffset("+offset.ioffset+")idxoffset("+offset.idxoffset+")");
		
		
		//$$$4. the coalesced blocks
		if (offset.yoffset!=0) {
			do2DShared(offset, memoryExpression);
			
		}
		else {
			do1DShared(offset, memoryExpression);
			
		}
		
	}


}


