package ece.ncsu.edu.gpucompiler.cuda.cetus;

import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.Expression;

public class MemoryExpression extends AssignmentExpression {
	

	int id;
	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	GLoop loop;
	MemoryArrayAccess lMemoryArrayAccess;
	MemoryArrayAccess rMemoryArrayAccess;
	
	public MemoryExpression(AssignmentExpression assignmentExpression, GLoop loop) throws UnsupportedCodeException {
		super((Expression)assignmentExpression.getLHS().clone(), assignmentExpression.getOperator(), (Expression)assignmentExpression.getRHS().clone());
		this.needs_parens = false;
		this.loop = loop;
		MemoryArray lma = parse(getLHS(), loop.getGProcedure());
		MemoryArray rma = parse(getRHS(), loop.getGProcedure());
		if (lma==null&&rma==null) {
			throw new IllegalArgumentException("this not memory statement: " + assignmentExpression);
		}
//		if (lma!=null&&rma!=null) throw new UnsupportedCodeException("only support statement with one memory access at lhs or rhs");
		if (lma!=null) {
			lMemoryArrayAccess = new MemoryArrayAccess((ArrayAccess)getLHS(), this);
			getLHS().swapWith(lMemoryArrayAccess);
		}
		if (rma!=null) {
			rMemoryArrayAccess = new MemoryArrayAccess((ArrayAccess)getRHS(), this);
			getRHS().swapWith(rMemoryArrayAccess);
		}		
	}
	

	
	public static boolean isMemoryExpression(AssignmentExpression assignmentExpression, GProcedure proc) {
		MemoryArray lma = parse(assignmentExpression.getLHS(), proc);
		MemoryArray rma = parse(assignmentExpression.getRHS(), proc);
		if (lma!=null||rma!=null) return true;
		return false;
	}
	
	public static MemoryArray parse(Expression ex, GProcedure proc) {
		if (ex instanceof ArrayAccess) {
			ArrayAccess aa = (ArrayAccess)ex;
			String name = aa.getArrayName().toString();
			MemoryArray ma = proc.getMemoryArray(name);
			return ma;

		}
		return null;
	}



	public GLoop getLoop() {
		return loop;
	}

	public void setLoop(GLoop loop) {
		this.loop = loop;
	}



	public MemoryArrayAccess getGlobalMemoryArrayAccess() {
		if (lMemoryArrayAccess!=null&&lMemoryArrayAccess.getMemoryArray().getMemoryType()==MemoryArray.MEMORY_GLOBAL) {
			return lMemoryArrayAccess;
		}
		if (rMemoryArrayAccess!=null&&rMemoryArrayAccess.getMemoryArray().getMemoryType()==MemoryArray.MEMORY_GLOBAL) {
			return rMemoryArrayAccess;
		}
		return null;
	}

	public MemoryArrayAccess getSharedMemoryArrayAccess() {
		if (lMemoryArrayAccess!=null&&lMemoryArrayAccess.getMemoryArray().getMemoryType()==MemoryArray.MEMORY_SHARED) {
			return lMemoryArrayAccess;
		}
		if (rMemoryArrayAccess!=null&&rMemoryArrayAccess.getMemoryArray().getMemoryType()==MemoryArray.MEMORY_SHARED) {
			return rMemoryArrayAccess;
		}
		return null;
	}
	
	
	public MemoryArrayAccess getlMemoryArrayAccess() {
		return lMemoryArrayAccess;
	}

	public MemoryArrayAccess getrMemoryArrayAccess() {
		return rMemoryArrayAccess;
	}

	public String toInfoString() {
		String s = "memory:"+toString()+
		";lMemoryArrayAccess="+lMemoryArrayAccess+
		";rMemoryArrayAccess="+rMemoryArrayAccess;
		return s;
	}
	

	public Object clone() {
		MemoryExpression me;
		try {
			me = new MemoryExpression((AssignmentExpression)super.clone(), loop);
			me.id = this.id;
			return me;
		} catch (UnsupportedCodeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return (AssignmentExpression)super.clone();
	}
}
