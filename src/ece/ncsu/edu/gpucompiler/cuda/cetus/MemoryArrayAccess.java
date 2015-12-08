package ece.ncsu.edu.gpucompiler.cuda.cetus;

import java.util.ArrayList;

import cetus.hir.ArrayAccess;
import cetus.hir.Expression;
import ece.ncsu.edu.gpucompiler.cuda.index.Address;

/**
 * memory array access
 * @author jack
 *
 */
public class MemoryArrayAccess extends ArrayAccess {
	
	
	Address[] indexs = new Address[3];	// low dimension
	
	MemoryExpression memoryExpression;
	String name;
	MemoryArray memoryArray;
	
	public MemoryArrayAccess(ArrayAccess arrayAccess, MemoryExpression memoryExpression) throws UnsupportedCodeException {
		super((Expression)arrayAccess.getArrayName().clone(), (Expression)arrayAccess.getIndex(0).clone());
		for (int i=1; i<arrayAccess.getNumIndices(); i++) {
			addIndex((Expression)arrayAccess.getIndex(i).clone());
		}
		this.name = getArrayName().toString();
		this.memoryExpression = memoryExpression;
		load();
	}

	
	
	public MemoryExpression getMemoryExpression() {
		return memoryExpression;
	}



	public void setMemoryExpression(MemoryExpression memoryExpression) {
		this.memoryExpression = memoryExpression;
	}




	public Address getIndexAddress(int dimension) {
		return indexs[dimension];
	}
	public void setIndexAddress(int dimension, Address index) {
		indexs[dimension] = index;
	}

	public void load() throws UnsupportedCodeException {
		for (int i=0; i<getNumIndices(); i++) {
			Expression ex = getIndex(i);
			indexs[i] = Address.parseAddress(ex, memoryExpression.getLoop());
		}
		memoryArray = memoryExpression.getLoop().getGProcedure().getMemoryArray(name);
	}	
	
	public void save() {
		ArrayList<Expression> list = new ArrayList<Expression>();
		for (int i=0; i<getNumIndices(); i++) {
			list.add(indexs[i].toExpression());
		}
		setIndices(list);
	}



	public MemoryArray getMemoryArray() {
		return memoryArray;
	}



	public void setMemoryArray(MemoryArray memoryArray) {
		this.memoryArray = memoryArray;
	}
		
	
	public Address getX() {
		Address x=null; 
		if (getNumIndices()==2) { // 2D
			x = getIndexAddress(1);
		}
		else {
			x = getIndexAddress(0);			
		}
		return x;		
	}
	
	public Address getY() {
		Address y=null; 
		if (getNumIndices()==2) { // 2D
			y = getIndexAddress(0);
		}
		return y;		
	}
	
//	public Object clone() {
//		MemoryArrayAccess maa;
//		try {
//			maa = new MemoryArrayAccess((ArrayAccess)super.clone());
//			return maa;
//		} catch (UnsupportedCodeException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		
//		return null;
//	}	
		
}
