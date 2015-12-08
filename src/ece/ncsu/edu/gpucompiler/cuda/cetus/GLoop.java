package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;
import java.util.List;

import cetus.analysis.LoopTools;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.Statement;

/**
 * @author jack
 * loop body support, only support Canonical Loop
 * 
 *
 */
public class GLoop extends ForLoop {

	long start;
	Expression end;
	IntegerLiteral increment;
	BinaryOperator incType;
	
	Identifier iterator;
	GProcedure func;
	
	List<Identifier> iterators = new ArrayList<Identifier>();
	
	GLoop parent =  null;
	
	public GLoop getLoopParent() {
		return parent;
	}

	public void setLoopParent(GLoop parent) {
		this.parent = parent;
		resetIterators();
	}

	void resetIterators() {
		iterators.clear();
		if (parent!=null) iterators.addAll(parent.iterators);
	}
	
	public List<Identifier> getIterators() {
		return iterators;
	}

	public GProcedure getGProcedure() {
		return func;
	}

	public void setGProcedure(GProcedure func) {
		this.func = func;
	}


	public CompoundStatement getCompoundStatementBody() {
		return (CompoundStatement)getBody();
	}
	
	public GLoop(ForLoop loop, GProcedure func) throws UnsupportedCodeException {
		super((Statement)loop.getInitialStatement().clone(), (Expression)loop.getCondition().clone(), (Expression)loop.getStep().clone(), (Statement)loop.getBody().clone());
		
		loop = this;
//		if (!LoopTools.isCanonical(loop)) {
//			throw new UnsupportedCodeException("we only support canonical loop: "+loop);
//		}
		
		Statement body = getBody();
		if (!(body instanceof CompoundStatement)) {
			CompoundStatement cs = new CompoundStatement();
			cs.addStatement((Statement)body.clone());
			cs.swapWith(body);
		}
		
		
		
		this.func = func;
		start = LoopTools.getLowerBound(loop);
		iterator = (Identifier)LoopTools.getIndexVariable(loop);
		resetIterators();
		
		{
			BinaryExpression cond = (BinaryExpression)loop.getCondition();
			Identifier id = (Identifier)cond.getLHS();
			if (!id.equals(iterator))
				throw new UnsupportedCodeException("not standard loop");
			end = cond.getRHS();
		}		
		
		{
			AssignmentExpression lstep = (AssignmentExpression)loop.getStep();
			Identifier id = (Identifier)lstep.getLHS();
			if (!id.equals(iterator))
				throw new UnsupportedCodeException("not standard loop");
			
			if (lstep.getOperator()!=AssignmentOperator.NORMAL) 
				throw new UnsupportedCodeException("operator of step must be '"+AssignmentOperator.NORMAL+"'");
			
			BinaryExpression be = (BinaryExpression)lstep.getRHS();
//			Identifier lid = (Identifier)be.getLHS();
			if (!(be.getLHS().equals(iterator)))
				throw new UnsupportedCodeException("not standard loop");			
			incType = (BinaryOperator)be.getOperator();
			if (!BinaryOperator.ADD.equals(incType)&&!BinaryOperator.MULTIPLY.equals(incType))
				throw new UnsupportedCodeException("not standard loop");
			if (!(be.getRHS() instanceof IntegerLiteral))
				throw new UnsupportedCodeException("not standard loop");
			IntegerLiteral rid = (IntegerLiteral)be.getRHS();
			increment = rid;
		}
		
		
	}
//	
//	protected void initMemoryExpression() {
//		
//	}
	

	

	
	public void save() {
		
	}
	
	public long getStart() {
		return start;
	}
	public void setStart(long start) {
		this.start = start;
	}
	
	public int getEndValue() {
		if (end instanceof IntegerLiteral) return (int)((IntegerLiteral)end).getValue();
		return Integer.MAX_VALUE;
	}
	
	public Expression setEndValue(Expression ex) {
		BinaryExpression cond = (BinaryExpression)getCondition();
		Identifier id = (Identifier)cond.getLHS();
		if (!id.equals(iterator))
			throw new UnsupportedOperationException("not standard loop");
		BinaryOperator op = cond.getOperator();
		end = cond.getRHS();		
		return new BinaryExpression(id, op, ex);
	}	
	
	public Expression getEnd() {
		return end;
	}
	public void setEnd(Expression end) {
		this.end = end;
	}

	public BinaryOperator getIncType() {
		return incType;
	}

	public void setIncType(BinaryOperator incType) {
		this.incType = incType;
	}

	public IntegerLiteral getIncrement() {
		return increment;
	}

	public void setIncrement(IntegerLiteral increment) {
		AssignmentExpression lstep = (AssignmentExpression)getStep();
		BinaryExpression be = new BinaryExpression((Identifier)getIterator().clone(), incType, increment);
		lstep.setRHS(be);
		this.increment = increment;
	}

	public Identifier getIterator() {
		return iterator;
	}
	public void setIterator(Identifier iterator) {
		this.iterator = iterator;
		resetIterators();
	}


	
	public int count(int defaultValue) {
		if (!(end instanceof IntegerLiteral)) return defaultValue;
		else {
			IntegerLiteral il = (IntegerLiteral)end;
			return (int)((il.getValue() - start)/increment.getValue());
		}
	}
	
	public String toInfoString() {
		String s = "start="+start+
		";end="+end+
		";increment="+increment+
		";incType="+incType+
		";iterator="+iterator;
		return s;
	}

	public Expression getStartIL() {
		// TODO Auto-generated method stub
		return new IntegerLiteral(start);
	}
	
	public boolean isSimpleLoop() {
		if (getStart()==0&&getIncrement().getValue()==1) {
			Expression ex = getEnd();
			if (ex instanceof IntegerLiteral) {
				if (((IntegerLiteral) ex).getValue()==1) {
					return true;
				}
			}
		}		
		return false;
	}
	
	public boolean isCoalescedLoop(int coalescedNumber) {
		if (getStart()==0&&getIncrement().getValue()==1) {
			Expression ex = getEnd();
			if (ex instanceof IntegerLiteral) {
				if (((IntegerLiteral) ex).getValue()==coalescedNumber) {
					return true;
				}
			}
		}		
		return false;
	}	
}
